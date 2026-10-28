"""Local, single-worker web adapter for the existing terminal agent."""
import json
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI, APIConnectionError, APIStatusError, APITimeoutError, RateLimitError
from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent import run_turn

TTL_SECONDS = 3600
MAX_TURNS = 12
MAX_HISTORY_CHARS = 120_000
MAX_CONVERSATIONS = 100


def serialized(history):
    return json.dumps(history, default=lambda item: item.model_dump(), ensure_ascii=False)


def trim_turns(turns):
    # Remove entire user turns, never individual function calls/results.
    turns = list(turns)
    while turns and (len(turns) > MAX_TURNS or len(serialized(sum(turns, []))) > MAX_HISTORY_CHARS):
        turns.pop(0)
    return turns


def sources_from(history):
    sources = {}
    for item in history:
        if isinstance(item, dict) and item.get('type') == 'function_call_output':
            try:
                result = json.loads(item['output'])
            except (ValueError, KeyError):
                continue
            if isinstance(result, list):
                for source in result:
                    if isinstance(source, dict) and {'source', 'section', 'text'} <= source.keys():
                        sources[source['source']] = dict(label=source['source'], section=source['section'], text=source['text'])
    return sources


@dataclass
class Conversation:
    turns: list = field(default_factory=list)
    search_number: int = 0
    citation_numbers: dict = field(default_factory=dict)
    touched: float = field(default_factory=time.monotonic)
    lock: threading.Lock = field(default_factory=threading.Lock)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    conversation_id: UUID | None = None
    message: str = Field(min_length=1, max_length=8000)

    @field_validator('message')
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError('Message must contain text.')
        return value.strip()


class Source(BaseModel):
    label: str
    section: str
    text: str
    citation_number: int | None = None
    cited: bool = False


class ChatResponse(BaseModel):
    conversation_id: UUID
    answer: str
    sources: list[Source]


def create_app(turn_runner=run_turn, client_factory=None):
    app = FastAPI(title='Ragas Paper Chat', description='Local agentic RAG demo. Conversations expire after one idle hour.')
    conversations = {}
    registry_lock = threading.Lock()
    app.state.conversations = conversations
    client_factory = client_factory or (lambda: OpenAI(timeout=45.0, max_retries=0))

    def acquire(conversation_id=None):
        with registry_lock:
            now = time.monotonic()
            for key, value in list(conversations.items()):
                if now - value.touched > TTL_SECONDS and not value.lock.locked():
                    del conversations[key]
            if conversation_id is None:
                if len(conversations) >= MAX_CONVERSATIONS:
                    raise HTTPException(503, 'Conversation capacity reached. Try again later.')
                conversation_id = str(uuid4())
                conversations[conversation_id] = Conversation()
            conversation = conversations.get(str(conversation_id))
            if conversation is None:
                raise HTTPException(404, 'Conversation expired or was cleared. Start a new chat.')
            if not conversation.lock.acquire(blocking=False):
                raise HTTPException(409, 'This conversation is busy. Wait for the current reply.')
            return str(conversation_id), conversation

    @app.post('/chat', response_model=ChatResponse)
    def chat(body: ChatRequest):
        key, conversation = acquire(body.conversation_id)
        committed = False
        try:
            turns = trim_turns(conversation.turns)
            history = sum(turns, [])
            with client_factory() as client:
                answer, updated, counter = turn_runner(client, history, body.message, conversation.search_number)
            all_sources = sources_from(updated)
            new_sources = sources_from(updated[len(history):])
            # Include passages cited from older turns, too.
            labels = list(dict.fromkeys(re.findall(r'\[(search\d+:\d+)\]', answer)))
            numbers = dict(conversation.citation_numbers)
            # Number only cited passages, in first-citation order. Keep numbers
            # stable across turns without changing the model's internal labels.
            selected = {}
            for label in labels:
                if label in all_sources:
                    numbers.setdefault(label, len(numbers) + 1)
                    selected[label] = {**all_sources[label], 'citation_number': numbers[label], 'cited': True}
            for label, source in new_sources.items():
                if label not in selected:
                    selected[label] = {**source, 'citation_number': numbers.get(label), 'cited': False}
            result = ChatResponse(conversation_id=key, answer=answer, sources=list(selected.values()))
            conversation.turns = trim_turns(turns + [updated[len(history):]])
            conversation.search_number = counter
            conversation.citation_numbers = numbers
            committed = True
            return result
        except RateLimitError:
            raise HTTPException(429, 'Model rate limit or quota reached. Please try again later.') from None
        except (APITimeoutError, APIConnectionError):
            raise HTTPException(504, 'The model service could not be reached. Please retry.') from None
        except APIStatusError:
            raise HTTPException(502, 'The model service rejected the request. Check backend credentials and model access.') from None
        except Exception:
            raise HTTPException(503, 'Unable to complete this turn. Check backend configuration and try again.') from None
        finally:
            with registry_lock:
                conversation.touched = time.monotonic()
                if body.conversation_id is None and not committed:
                    conversations.pop(key, None)
                conversation.lock.release()

    @app.delete('/conversations/{conversation_id}', status_code=204)
    def delete_conversation(conversation_id: UUID):
        key, conversation = acquire(conversation_id)
        try:
            with registry_lock:
                del conversations[key]
        finally:
            conversation.lock.release()
        return Response(status_code=204)

    static = Path(__file__).resolve().parent / 'static'
    if static.is_dir():
        app.mount('/static', StaticFiles(directory=static), name='static')

        @app.get('/', include_in_schema=False)
        def index():
            return FileResponse(static / 'index.html')
    return app


app = create_app()
