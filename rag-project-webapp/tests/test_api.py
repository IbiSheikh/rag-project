import json
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from api import create_app, trim_turns, sources_from, MAX_HISTORY_CHARS
from agent import run_turn


def runner(client, history, question, counter):
    updated = history + [{'role': 'user', 'content': question}]
    if question != 'hello':
        counter += 1
        updated += [{'type': 'function_call', 'call_id': str(counter)}, {'type': 'function_call_output', 'call_id': str(counter), 'output': json.dumps([{'source': f'search{counter}:1', 'section': 'Faithfulness', 'text': 'Claims supported by context.'}])}]
    answer = 'Hello!' if question == 'hello' else f'Evidence [search{counter}:1]'
    return answer, updated + [{'role': 'assistant', 'content': answer}], counter


def setup(r=runner):
    app = create_app(r, lambda: nullcontext(None))
    return app, TestClient(app)


def test_static_docs_and_chat_routes():
    _, client = setup()
    for path in ['/', '/docs', '/openapi.json', '/static/app.js', '/static/styles.css']:
        assert client.get(path).status_code == 200
    assert client.get('/health').status_code == 404
    assert set(client.get('/openapi.json').json()['paths']) == {'/chat', '/conversations/{conversation_id}'}
    assert client.get('/.env').status_code == 404


def test_greeting_search_followup_isolation_delete():
    app, client = setup()
    first = client.post('/chat', json={'message': 'hello'}).json()
    key = first['conversation_id']
    assert first['sources'] == []
    paper = client.post('/chat', json={'conversation_id': key, 'message': 'faithfulness'}).json()
    assert paper['sources'][0]['label'] == 'search1:1'
    followup = client.post('/chat', json={'conversation_id': key, 'message': 'explain more'}).json()
    assert followup['sources'][0]['label'] == 'search2:1'
    assert len(app.state.conversations[key].turns) == 3
    other = client.post('/chat', json={'message': 'faithfulness'}).json()
    assert other['conversation_id'] != key
    assert other['sources'][0]['label'] == 'search1:1'
    assert client.delete('/conversations/' + key).status_code == 204
    assert client.post('/chat', json={'conversation_id': key, 'message': 'hello'}).status_code == 404


@pytest.mark.parametrize('body', [{'message': ''}, {'message': '  '}, {'message': 'x'*8001}, {'message':'hi','conversation_id':'bad'}, {'message': 'hi', 'extra':1}])
def test_validation(body):
    _, client = setup()
    assert client.post('/chat', json=body).status_code == 422


def test_failure_rollback_and_redaction():
    app, client = setup()
    key = client.post('/chat', json={'message':'hello'}).json()['conversation_id']
    before = list(app.state.conversations[key].turns)
    def fail(*args):
        raise RuntimeError('SECRET-SENTINEL')
    app2, failing = setup(fail)
    assert failing.post('/chat', json={'message':'hi'}).status_code == 503
    assert not app2.state.conversations
    # An existing conversation must survive a failed turn unchanged.
    app2.state.conversations[key] = app.state.conversations[key]
    response = failing.post('/chat', json={'conversation_id':key,'message':'hi'})
    assert 'SECRET-SENTINEL' not in response.text
    assert app2.state.conversations[key].turns == before


def test_expiry_and_busy():
    app, client = setup()
    key = client.post('/chat', json={'message':'hello'}).json()['conversation_id']
    conv = app.state.conversations[key]
    conv.lock.acquire()
    try:
        assert client.post('/chat', json={'conversation_id':key,'message':'hi'}).status_code == 409
        assert client.delete('/conversations/'+key).status_code == 409
    finally:
        conv.lock.release()
    conv.touched -= 3601
    assert client.post('/chat', json={'conversation_id':key,'message':'hi'}).status_code == 404


def test_real_overlapping_requests():
    entered, release = threading.Event(), threading.Event()
    def slow(*args):
        entered.set()
        assert release.wait(5)
        return runner(*args)
    app, client = setup()
    key = client.post('/chat', json={'message':'hello'}).json()['conversation_id']
    slow_app, slow_client = setup(slow)
    slow_app.state.conversations[key] = app.state.conversations[key]
    with ThreadPoolExecutor() as pool:
        pending = pool.submit(slow_client.post, '/chat', json={'conversation_id':key,'message':'hello'})
        assert entered.wait(5)
        try:
            assert slow_client.post('/chat', json={'conversation_id':key,'message':'hi'}).status_code == 409
        finally:
            release.set()
        assert pending.result().status_code == 200


def test_trim_preserves_turn_pairs():
    turns = []
    for n in range(15):
        _, history, _ = runner(None, [], 'paper', n)
        turns.append(history)
    kept = trim_turns(turns)
    assert kept == turns[-12:]
    assert all(t[1]['call_id'] == t[2]['call_id'] for t in kept)
    assert trim_turns([[{'role':'user','content':'x'*(MAX_HISTORY_CHARS+1)}]]) == []


def test_old_citations_return_sources():
    app, client = setup()
    key = client.post('/chat', json={'message':'paper'}).json()['conversation_id']
    def old(client, history, question, number):
        return 'Earlier [search1:1]', history + [{'role':'user','content':question}], number
    other, c = setup(old)
    other.state.conversations[key] = app.state.conversations[key]
    assert c.post('/chat', json={'conversation_id':key,'message':'followup'}).json()['sources'][0]['label'] == 'search1:1'


def test_agent_tool_loop_and_insufficient_evidence(monkeypatch):
    import agent
    client = Mock()
    call = SimpleNamespace(type='function_call', name='search_documents', arguments='{"query":"paper evidence"}', call_id='c1')
    client.responses.create.side_effect = [SimpleNamespace(output=[call], output_text=''), SimpleNamespace(output=[], output_text='The retrieved passages do not provide enough information.')]
    monkeypatch.setattr(agent, 'search_documents', lambda q,n: [])
    original = []
    answer, history, number = run_turn(client, original, 'unknown', 0)
    assert 'retrieved passages' in answer
    assert original == [] and number == 1
    assert history[-1]['call_id'] == 'c1'
    assert all(call.kwargs['model'] == 'gpt-6-luna' and call.kwargs['store'] is False for call in client.responses.create.call_args_list)


def test_agent_six_round_limit(monkeypatch):
    import agent
    client = Mock()
    client.responses.create.return_value = SimpleNamespace(output=[SimpleNamespace(type='function_call', name='search_documents', arguments='{"query":"q"}', call_id='c')], output_text='')
    monkeypatch.setattr(agent, 'search_documents', lambda q,n: [])
    with pytest.raises(RuntimeError):
        run_turn(client, [], 'question', 0)
    assert client.responses.create.call_count == 6
    assert client.responses.create.call_args.kwargs['tool_choice'] == 'none'

@pytest.mark.parametrize('kind,status', [('rate',429),('timeout',504),('connection',504),('status',502)])
def test_api_error_mapping(kind, status):
    import httpx
    from openai import RateLimitError, APITimeoutError, APIConnectionError, APIStatusError
    request = httpx.Request('POST','https://example.test')
    response = httpx.Response(403, request=request)
    errors = {
        'rate': lambda: RateLimitError('private', response=response, body=None),
        'timeout': lambda: APITimeoutError(request=request),
        'connection': lambda: APIConnectionError(request=request),
        'status': lambda: APIStatusError('private', response=response, body=None),
    }
    def fail(*args):
        raise errors[kind]()
    _, client = setup(fail)
    result = client.post('/chat', json={'message':'hello'})
    assert result.status_code == status
    assert 'private' not in result.text


def test_capacity(monkeypatch):
    import api
    monkeypatch.setattr(api, 'MAX_CONVERSATIONS', 1)
    _, client = setup()
    assert client.post('/chat', json={'message':'hello'}).status_code == 200
    assert client.post('/chat', json={'message':'hello'}).status_code == 503


def test_oversized_success_keeps_conversation(monkeypatch):
    import api
    monkeypatch.setattr(api, 'MAX_HISTORY_CHARS', 1)
    app, client = setup()
    result = client.post('/chat', json={'message':'hello'})
    assert result.status_code == 200
    key = result.json()['conversation_id']
    assert key in app.state.conversations
    assert app.state.conversations[key].turns == []
    assert client.post('/chat', json={'conversation_id':key,'message':'hello'}).status_code == 200


def test_agent_greeting_no_search(monkeypatch):
    import agent
    search = Mock()
    monkeypatch.setattr(agent, 'search_documents', search)
    client = Mock()
    client.responses.create.return_value = SimpleNamespace(output=[], output_text='Hello!')
    answer, history, counter = run_turn(client, [], 'hello', 0)
    assert answer == 'Hello!' and counter == 0
    search.assert_not_called()


def test_agent_search_failure_becomes_tool_result(monkeypatch):
    import agent
    monkeypatch.setattr(agent, 'search_documents', Mock(side_effect=RuntimeError('SECRET-SENTINEL')))
    client = Mock()
    call = SimpleNamespace(type='function_call', name='search_documents', arguments='{"query":"q"}', call_id='failed')
    client.responses.create.side_effect = [SimpleNamespace(output=[call], output_text=''), SimpleNamespace(output=[], output_text='Retrieved evidence is unavailable.')]
    _, history, _ = run_turn(client, [], 'paper', 0)
    result = json.loads(history[-1]['output'])
    assert 'error' in result
    assert 'SECRET-SENTINEL' not in history[-1]['output']


def test_followup_receives_full_prior_history():
    observed = []
    def capture(*args):
        observed.append(list(args[1]))
        return runner(*args)
    app, client = setup(capture)
    key = client.post('/chat', json={'message':'paper'}).json()['conversation_id']
    prior = list(app.state.conversations[key].turns[0])
    client.post('/chat', json={'conversation_id':key,'message':'followup'})
    assert observed == [[], prior]


def test_citation_numbers_follow_first_use_and_persist():
    def numbered(client, history, question, counter):
        counter += 1
        results = [{'source':f'search{counter}:{i}', 'section':'**3** **Evaluation Strategies**', 'text':f'Passage {i}'} for i in range(1,4)]
        answer = '[search1:3] then [search1:1] and [search1:3]' if counter == 1 else '[search1:3] and [search2:2]'
        return answer, history + [{'role':'user','content':question}, {'type':'function_call_output','call_id':str(counter),'output':json.dumps(results)}], counter
    _, client = setup(numbered)
    first = client.post('/chat', json={'message':'paper'}).json()
    assert [(s['label'],s['citation_number'],s['cited']) for s in first['sources']] == [('search1:3',1,True),('search1:1',2,True),('search1:2',None,False)]
    second = client.post('/chat', json={'message':'followup','conversation_id':first['conversation_id']}).json()
    assert [(s['label'],s['citation_number']) for s in second['sources'] if s['cited']] == [('search1:3',1),('search2:2',3)]
    fresh = client.post('/chat', json={'message':'paper'}).json()
    assert fresh['sources'][0]['citation_number'] == 1


def test_missing_citation_has_no_fabricated_source():
    def missing(client, history, question, counter):
        return 'Claim [search99:1]', history + [{'role':'user','content':question}], counter
    app, client = setup(missing)
    result = client.post('/chat', json={'message':'paper'}).json()
    assert result['sources'] == []
    assert app.state.conversations[result['conversation_id']].citation_numbers == {}
