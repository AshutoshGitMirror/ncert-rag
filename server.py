"""OpenAI-compatible NCERT RAG API.

POST /v1/chat/completions  — Ask NCERT questions
GET  /v1/models            — List available model
GET  /health               — Health check

Usage:
  export GEMINI_API_KEY=...
  pip install -r requirements.txt
  uvicorn server:app --host 0.0.0.0 --port 8000
"""
import json, os, pickle, time, logging
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ncert-rag")

# --- Config ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
assert GEMINI_API_KEY, "GEMINI_API_KEY required"
EMBED_DIM = 768
INDEX_URL = os.environ.get("INDEX_URL", "")
client = genai.Client(api_key=GEMINI_API_KEY)

# --- Data ---
index: faiss.Index = None
chunks_meta: list = []

def download_index():
    """Download FAISS index + metadata if not present."""
    for name in ["faiss.index", "chunks_meta.pkl"]:
        path = Path(f"/data/{name}")
        if path.exists():
            log.info(f"Using cached {name}")
            continue
        url = f"{INDEX_URL}/{name}"
        log.info(f"Downloading {url}...")
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
        log.info(f"Downloaded {name} ({len(r.content)//1024} KB)")

def load_index():
    global index, chunks_meta
    index = faiss.read_index("/data/faiss.index")
    with open("/data/chunks_meta.pkl", "rb") as f:
        chunks_meta = pickle.load(f)
    log.info(f"Loaded index: {index.ntotal} vectors, {len(chunks_meta)} chunks")

if INDEX_URL:
    Path("/data").mkdir(exist_ok=True)
    download_index()
load_index()

# --- Embedding ---
def get_embedding(texts: list[str]) -> list[list[float]]:
    result = client.models.embed_content(
        model="gemini-embedding-2",
        contents=[t[:2000] for t in texts],
        config={"output_dimensionality": EMBED_DIM},
    )
    return [e.values for e in result.embeddings]

# --- RAG ---
def rag(query_text: str, top_k: int = 5) -> tuple[str, list]:
    q_emb = get_embedding([query_text])[0]
    q_vec = np.array([q_emb], dtype=np.float32)
    scores, indices = index.search(q_vec, top_k)
    context_parts = []
    for score, idx in zip(scores[0], indices[0]):
        m = chunks_meta[idx]
        context_parts.append(
            f"[Std {m['std']} | {m['subj']} | {m['book']} | "
            f"Ch {m['ch_num']}: {m['ch']} | Pages {m['pages']}]\n{m['text']}"
        )
    context = "\n---\n".join(context_parts)
    return context, [m["images"] for m in [chunks_meta[i] for i in indices[0]] if m["images"]]

# --- API Models ---
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "ncert-rag"
    messages: list[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1024
    stream: Optional[bool] = False

class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class Choice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"

class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completions"
    created: int
    model: str = "ncert-rag"
    choices: list[Choice]
    usage: Usage

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = 1710000000
    owned_by: str = "ncert-rag"

class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelInfo]

# --- API ---
app = FastAPI(title="NCERT RAG API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
def health():
    return {"status": "ok", "vectors": index.ntotal, "chunks": len(chunks_meta)}

@app.get("/v1/models")
def list_models():
    return ModelList(data=[ModelInfo(id="ncert-rag")])

@app.post("/v1/chat/completions")
def chat_completion(req: ChatRequest):
    # Extract the last user message as the question
    question = ""
    system_prompt = ""
    for m in req.messages:
        if m.role == "system":
            system_prompt = m.content
        elif m.role == "user":
            question = m.content

    if not question:
        raise HTTPException(400, "No user message found")

    # RAG
    context, imgs = rag(question)

    # Build prompt
    if not system_prompt:
        system_prompt = "You are a friendly tutor for school children. Answer based on NCERT textbook excerpts. Be clear, engaging, and age-appropriate. For each claim, cite which textbook and chapter it comes from."

    full_prompt = f"{system_prompt}\n\nQuestion: {question}\n\nTextbook excerpts:\n{context}"

    # Call Gemini
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=full_prompt,
    )

    answer = response.text

    # Token estimate
    p_tokens = len(full_prompt.split()) * 2
    c_tokens = len(answer.split()) * 2

    return ChatResponse(
        id=f"chatcmpl-{int(time.time())}",
        created=int(time.time()),
        choices=[Choice(message=ChatMessage(role="assistant", content=answer))],
        usage=Usage(prompt_tokens=p_tokens, completion_tokens=c_tokens, total_tokens=p_tokens + c_tokens),
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
