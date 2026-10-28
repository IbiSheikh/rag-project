# Ragas Paper Chat

FastAPI serves a small chat frontend and wraps the existing tool-calling agent.
All changes are in `rag-project-webapp`; the original project is untouched.

## Run (PowerShell)

```powershell
cd C:\Users\ibish\Desktop\rag-project-webapp
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000 for chat, or http://127.0.0.1:8000/docs for Swagger UI.
Keep OPENAI_API_KEY in the existing `.env`. The key is used exclusively by Python.
Do not run main.py for normal web startup: it is the indexing pipeline. The existing
`chroma_db/rag_documents` collection is reused; startup does not extract or index PDFs.
The first paper query loads SentenceTransformer and may download model files if they
are not already cached. Greetings do not load the embedding model.

Terminal mode still works:

```powershell
.\.venv\Scripts\python.exe agent.py
```

## Backend contract

- `POST /chat`: `{"conversation_id": null, "message": "How does the paper define faithfulness?"}`.
  Returns `conversation_id`, `answer`, and `sources` (each with `label`, `section`, `text`, `citation_number`, `cited`).
  Reuse the returned ID for follow-ups. Unknown/expired IDs return 404 rather than silently losing context.
- `DELETE /conversations/{id}`: removes a conversation; returns 204, or 404 if missing.
- Invalid messages return 422; overlapping requests or deletion during a turn return 409.
  Model rate/quota errors return 429, connection/timeouts 504, provider rejections 502,
  and other turn/configuration failures 503. Error bodies omit raw exception details.

Try POST /chat in `/docs` with `conversation_id: null`, copy the returned UUID into
another request, then try a follow-up. DELETE that UUID and confirm subsequent use returns 404.

## How state and sources work

`agent.run_turn(client, history, question, search_number)` keeps its three-value return
and original tool loop. An optional `verbose` keyword controls terminal diagnostics.
The model stays `gpt-6-luna`, uses `store=False`, and replays response items and tool results.
Encrypted reasoning content is requested so stateless reasoning items can be replayed.
Each model response has a 2,000 output-token limit and the sixth round disables tools.
See the official conversation-state guide: https://developers.openai.com/api/docs/guides/conversation-state.

`api.py` owns independent histories and search counters. A lock reserves each conversation
for a whole turn. Different conversations can run concurrently. Only successful turns commit.
History is retained as whole turns, capped at 12 turns and 120,000 serialized characters;
oldest whole turns are dropped first, preserving tool call/result pairs. A single oversized
turn is answered but not retained. This is a character budget, not an exact token count.
Search numbers never reset during trimming, so labels remain unique within a conversation.
Sources are extracted from actual tool outputs: all newly retrieved passages plus any older
passages explicitly cited in the answer. Returned source labels establish provenance, not
proof that every generated claim is supported; semantic citation accuracy still needs review.

Conversations expire after one idle hour. Cleanup runs on chat/delete requests; active turns
are not expired. Capacity is 100 conversations. Server restarts clear all histories.
Run **one worker**: this in-memory store is not shared between worker processes.
This is a local internship demo without user authentication; keep it bound to localhost.
For shared deployment, add authentication, shared state/locking, and request controls.

`static/app.js` retains only the conversation ID in localStorage; the transcript is not
restored after reload, though server context remains. Expired IDs clear on a 404 so the next
send starts a new conversation. New Chat deletes the old server conversation before resetting.
Answers and passages render a limited Markdown subset using DOM elements and text nodes. Raw HTML stays inert; full Markdown and LaTeX are not supported.
No API key or environment file is served: only the static directory is mounted.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Tests inject a fake model runner/client and require neither API calls nor embedding downloads.
They check greetings, sources, follow-up history, new-chat isolation, deletion, expiry,
concurrent requests, failed-turn rollback, validation, history limits, old citations,
provider error mapping, insufficient-evidence tool flow, and the six-round agent limit.
These deterministic checks do not establish real-model factual accuracy.

For an opt-in real model/index smoke run (requires the running server and incurs API usage):

```powershell
.\.venv\Scripts\python.exe tests/live_smoke.py
```

## Verification performed (2026-10-27)

- Fresh `.venv` installed; `pip check` reports no broken requirements.
- `python -m pytest -q`: **23 passed** (fake model/retrieval, no model charges).
- Live smoke script completed against the running server using `gpt-6-luna` and the existing index:
  greeting with no sources; faithfulness with five passages and citations; follow-up calculation
  with further searches; insufficient GPU/electricity evidence acknowledged; fresh conversation
  did not know prior questions; both smoke conversations deleted successfully.
- Spot-checked the live faithfulness definition and formula against returned source passages.
  This is not a systematic semantic citation evaluation.
- Browser: chat page served correctly, a submitted RAG question received a cited answer,
  loading controls returned to enabled, and the Abstract source expanded correctly.
- Swagger `/docs` was verified during initial implementation. The health endpoint used in that initial check has since been removed.
- New Chat isolation/deletion and provider error conditions were covered in automated tests;
  browser New Chat and every error display were not separately exercised manually.
- Sandboxed test/server starts hung; tests and the server succeeded outside the sandbox.

Preview: `preview.png` shows the working chat with an expanded source.


## Citation interface update (2026-10-28)

The API now returns `citation_number` (nullable integer) and `cited` (boolean) on
sources while preserving `label`, `section`, and `text`. Model history still uses
its internal search labels. The backend numbers cited passages in first-use order
and preserves their numbers across the conversation. The frontend displays linked
numbers, cleaned section titles, cited sources first, and a collapsed group for
other searched passages. Unknown citations explicitly show source unavailable.
The existing index has no page metadata, so the interface does not invent pages.

`static/render.js` builds a limited Markdown subset with DOM nodes (bold, emphasis,
headings, flat lists, code, and paragraphs). Raw HTML remains text. Citation clicks
open, focus, and highlight their passage. See ARCHITECTURE.md for the full workflow.

Validation: **25 backend tests passed**. Isolated browser checks with deterministic
fake answers verified formatting, clickable evidence, stable old citation numbers,
uncited-source grouping, missing citations, inert HTML, and New Chat. The model,
embedding configuration, and index were unchanged; no real model calls or indexing
were needed for this UI update. The temporary UI server used port 8001 and was stopped.

Restart your existing Uvicorn process and refresh the browser to load both API and UI
changes. Ordinary startup does not use `--reload`. Optional browser fixture:
`python -m tests.ui_server` (port 8001; fake model responses, no API calls).
