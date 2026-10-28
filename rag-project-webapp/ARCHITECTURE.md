# Explaining this application's architecture

## A short explanation you can give in a presentation

I built a conversational RAG application using a Python FastAPI backend and an
HTML, CSS, and JavaScript frontend. I extracted a research paper, split its text
into overlapping chunks, generated local embeddings with SentenceTransformer,
and stored those chunks and vectors in persistent ChromaDB. When someone asks a
question, an OpenAI model acts as an agent: it can request document searches,
inspect the returned passages, and search again before producing a cited answer.
FastAPI manages conversation state and returns the answer and evidence to the
browser, which presents numbered citations linked to expandable passages.

## Two distinct workflows

Document preparation (already completed for the current paper):

PDF -> PyMuPDF4LLM -> Markdown -> sections -> overlapping chunks
    -> SentenceTransformer embeddings -> persistent ChromaDB collection

Question answering (every user message):

Browser -> POST /chat -> FastAPI -> run_turn -> OpenAI Responses API
    -> optional search_documents tool request
    -> Python embeds search query -> ChromaDB finds nearest chunks
    -> Python sends passage text back as tool results -> model answers
    -> FastAPI returns JSON -> browser renders answer and linked evidence

A greeting can skip retrieval. A paper question should trigger search. An agent
turn can perform multiple searches. Embedding the query is not document indexing:
the stored paper vectors are reused, and the PDF is not extracted for each message.

## Technologies and reasons

| Component | Technology | Why it is used here |
| --- | --- | --- |
| Page structure | HTML | Defines the chat, input, buttons, and source panels. |
| Appearance | CSS | Handles layout, responsive sizing, answer typography, and citation highlighting. |
| Browser behavior | Vanilla JavaScript | Sends HTTP requests, handles loading/errors, and builds answer/source elements without a frontend build system. Appropriate for this small interface. |
| Web backend | Python + FastAPI | Wraps the existing Python agent and retrieval pipeline in HTTP endpoints. Pydantic validates requests; FastAPI provides interactive Swagger documentation. |
| Server process | Uvicorn | Runs the FastAPI ASGI application and accepts browser requests. |
| PDF extraction | PyMuPDF4LLM | Converts PDF content into Markdown that can retain headings and text structure. Extraction quality still affects retrieval quality. |
| Section detection | Regular expressions | Uses the existing paper heading patterns to preserve section metadata. This is tailored to the indexed paper rather than universal PDF layout handling. |
| Chunk splitting | LangChain RecursiveCharacterTextSplitter | Breaks large sections into manageable passages while preferring natural separators. The project uses 1,000-character chunks with 200-character overlap, not token counts. |
| Embeddings | SentenceTransformer all-MiniLM-L6-v2 | Produces normalized 384-dimensional representations locally for both stored chunks and queries. Using the same model puts both in the same vector space. |
| Vector store | Persistent ChromaDB | Stores vectors, text, and section metadata locally and supports similarity search without a separate database server. |
| Similarity measure | Cosine distance | Ranks chunks by how close their embedding directions are to the query. Lower distance means closer, not guaranteed factual relevance. |
| Answering model | gpt-6-luna through OpenAI Responses API | Interprets the conversation, requests tools, and writes an answer using retrieved evidence. Preserved because it works with this project's API access. |
| Agent orchestration | Plain Python tool loop | Explicitly handles model tool requests and results. Easy to inspect for an internship project; no separate agent framework is needed. |
| Conversation storage | In-memory Python objects and locks | Simple isolated chat histories and search counters; one active request per conversation prevents conflicting updates. This demo uses one worker and loses histories on restart. |
| Citation presentation | Backend numbering + safe DOM renderer | Keeps internal model labels stable while showing readable numbers, clean section names, and linked passages. DOM text nodes keep model/source HTML inert. |
| Verification | pytest + FastAPI TestClient | Checks API state, failure handling, tool loops, and citation numbering without requiring real model calls. |

## Exactly where RAG happens

Retrieval: embed the search query and fetch the top five similar chunks from ChromaDB.
Augmentation: put the retrieved text into the model's context as function tool results.
Generation: ask the model to answer using those passages and cite the provided labels.

The conversational path uses agent.py for this loop. augmentation.py and generation.py
belong to the separate standalone RAG path and are not called by POST /chat.

There are two model roles: SentenceTransformer creates embeddings and does not write
answers; gpt-6-luna requests searches and generates language. ChromaDB stores and searches
vectors; it is not a language model. FastAPI coordinates requests; JavaScript presents
results. Neither model is trained on the PDF in this application: the language model sees
selected passages in its request context. Those passages are sent to OpenAI for answering.

Agentic means the language model can decide when and how to search, including rewriting
follow-ups into standalone queries and making several searches before answering. The tool
executes in Python. The model does not connect directly to ChromaDB or execute arbitrary
Python. The loop has six model rounds; the final round disables tools to force completion.

## Example: asking about faithfulness

1. JavaScript sends the question and conversation ID to POST /chat.
2. FastAPI finds the conversation and reserves its lock.
3. The Python agent sends history, instructions, and the tool schema to the model.
4. The model requests search_documents with a query about faithfulness.
5. Python embeds that query and retrieves up to five passages from the stored paper.
6. Python labels those passages and sends the tool results back to the model.
7. The model can search again or produce an answer with internal source labels.
8. FastAPI extracts actual tool passages and assigns stable display numbers to cited ones.
9. The browser renders the answer. Clicking [1] expands, focuses, and highlights its evidence.
10. Successful history is saved for follow-ups; failed turns do not overwrite it.

## Citation design and its limits

Internal labels such as search2:3 mean passage 3 of search 2. They remain in the model
history so existing citations and tool results stay consistent. The API adds cited and
citation_number fields to each returned source. Only cited passages receive a new display
number. Numbering persists across turns, resets for a new conversation, and can have gaps
within one answer because it spans the conversation. The frontend uses these fields rather
than inventing a source for any generated label. Missing labels show 'source unavailable'.

Cited sources appear first; uncited retrieval results are collapsed under 'Other passages
searched'. Repeated sections can appear because multiple chunks can come from one section.
Two identical passages from different searches can receive separate citation numbers: the
identity currently follows the tool-result label rather than a stable document chunk ID.
Page numbers are not shown because the existing index stores section labels but no reliable
page metadata. Adding pages would require a deliberate extraction/index metadata update.

The renderer supports a limited Markdown subset: paragraphs, bold, emphasis, headings,
flat lists, and fenced/inline code. It does not execute HTML or load model-generated images.
Full Markdown tables, nested formatting, links, and LaTeX are not rendered.

A valid clickable citation proves that a retrieved passage exists, not that every associated
claim follows from it. Semantic citation support still needs evaluation. Also, this project
answers questions about the Ragas paper; it does not yet use the Ragas evaluation library
to automatically score its own answers.
