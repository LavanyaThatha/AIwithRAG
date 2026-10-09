"""Step 2: retrieve relevant chunks and ask an LLM to answer from them.

Default backend: Ollama (free, local, no API key).
Optional backend: Claude API  ->  set LLM_BACKEND=claude and ANTHROPIC_API_KEY.
"""
import os

import chromadb

DB_DIR = "chroma_db"
COLLECTION = "docs"
TOP_K = 4

LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama").lower()      # "ollama" or "claude"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the provided context. "
    "If the answer is not in the context, say \"I couldn't find that in the documents.\" "
    "Do not make things up. Mention the source numbers like [1] or [2] that support your answer."
)


def get_collection():
    client = chromadb.PersistentClient(path=DB_DIR)
    return client.get_collection(COLLECTION)   # raises if ingest.py was never run


def retrieve(question: str, k: int = TOP_K):
    res = get_collection().query(query_texts=[question], n_results=k)
    hits = []
    for text, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        hits.append({"text": text, "source": meta["source"], "page": meta["page"], "distance": dist})
    return hits


def _ask_ollama(messages):
    import ollama
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
    )
    return response["message"]["content"]


def _ask_claude(messages):
    import anthropic
    client = anthropic.Anthropic()             # reads ANTHROPIC_API_KEY
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=800,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    return response.content[0].text

GREETINGS = {"hi", "hello", "hey", "hii", "hola", "good morning", "good afternoon", "good evening"}


def is_greeting(text: str) -> bool:
    return text.strip().lower().strip("!.?, ") in GREETINGS


def answer(question: str, history=None):
    """history: list of {"role": "user"|"assistant", "content": str} from earlier turns."""
    if is_greeting(question):
        return "Hello! 👋 Ask me anything about your documents.", []
    
    hits = retrieve(question)
    context = "\n\n".join(
        f"[{i + 1}] (source: {h['source']}, page {h['page']})\n{h['text']}"
        for i, h in enumerate(hits)
    )
    user_msg = f"Context:\n{context}\n\nQuestion: {question}"

    messages = list(history or [])[-6:]
    messages.append({"role": "user", "content": user_msg})

    reply = _ask_claude(messages) if LLM_BACKEND == "claude" else _ask_ollama(messages)
    return reply, hits