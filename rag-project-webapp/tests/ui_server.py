"""Deterministic browser fixture. Run from project root: python -m tests.ui_server.
Uses the real frontend/backend with fake agent responses; never calls OpenAI.
"""
import json
from contextlib import nullcontext
import uvicorn
from api import create_app


def demo(client, history, question, counter):
    counter += 1
    sources = [
        {'source':f'search{counter}:1','section':'**3** **Evaluation Strategies**','text':'**Faithfulness** measures whether claims follow from the supplied context.\n\n<script>alert("unsafe")</script>'},
        {'source':f'search{counter}:2','section':'**6** **Conclusions**','text':'The paper introduces reference-free evaluation.'},
        {'source':f'search{counter}:3','section':'**1** **Introduction**','text':'Additional background, retrieved but not cited.'},
    ]
    answer = f'## Evaluation overview\n\nThe paper evaluates **faithfulness** and **answer relevance**. [search{counter}:2]\n\n- **Faithfulness:** Are claims supported by context? [search{counter}:1]\n- Answer relevance: Does the answer address the question?\n\n<script>alert("unsafe")</script>\n\nMissing evidence [search99:1]'
    if question.lower() != 'safety':
        answer = answer.split('\n\n<script>')[0]
        sources[0]['text'] = sources[0]['text'].split('\n\n<script>')[0]
    if counter > 1:
        answer += '\n\nEarlier evidence [search1:2].'
    updated = history + [{'role':'user','content':question}, {'type':'function_call_output','call_id':str(counter),'output':json.dumps(sources)}]
    return answer, updated, counter


if __name__ == '__main__':
    uvicorn.run(create_app(demo, lambda: nullcontext(None)), host='127.0.0.1', port=8001)

