"""Opt-in real model/index smoke test; incurs OpenAI usage. Never logs credentials."""
import json
import httpx


def main():
    with httpx.Client(base_url='http://127.0.0.1:8000', timeout=330) as client:
        def ask(message, conversation_id=None):
            response = client.post('/chat', json={'conversation_id': conversation_id, 'message': message})
            response.raise_for_status()
            data = response.json()
            print(json.dumps({'message': message, **data}, ensure_ascii=True), flush=True)
            return data
        greeting = ask('Hello!')
        assert not greeting['sources']
        paper = ask('How does the paper define faithfulness?', greeting['conversation_id'])
        assert paper['sources']
        ask('How is that metric calculated?', paper['conversation_id'])
        ask('What exact GPU model and electricity cost did the authors use for training?', paper['conversation_id'])
        isolated = ask('What was my previous question?')
        assert isolated['conversation_id'] != paper['conversation_id']
        for key in [paper['conversation_id'], isolated['conversation_id']]:
            assert client.delete('/conversations/' + key).status_code == 204
        print('Live smoke checks completed.', flush=True)


if __name__ == '__main__':
    main()
