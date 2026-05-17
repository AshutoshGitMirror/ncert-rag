import json, os, pickle, time, logging, asyncio
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from google import genai
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ncert-rag")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
assert GEMINI_API_KEY, "GEMINI_API_KEY required"
EMBED_DIM = 768
INDEX_URL = os.environ.get("INDEX_URL", "")
client = genai.Client(api_key=GEMINI_API_KEY)

index: faiss.Index = None
chunks_meta: list = []

def download_index():
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
    vectors = index.reconstruct_n(0, index.ntotal)
    faiss.normalize_L2(vectors)
    norm_index = faiss.IndexFlatIP(EMBED_DIM)
    norm_index.add(vectors)
    index = norm_index
    log.info(f"Loaded+normalized index: {index.ntotal} vectors, {len(chunks_meta)} chunks")

if INDEX_URL:
    Path("/data").mkdir(exist_ok=True)
    download_index()
load_index()

def get_embedding(texts: list[str]) -> list[list[float]]:
    result = client.models.embed_content(
        model="gemini-embedding-2",
        contents=[t[:2000] for t in texts],
        config={"output_dimensionality": EMBED_DIM},
    )
    return [e.values for e in result.embeddings]

def rag(query_text: str, top_k: int = 5) -> str:
    q_emb = get_embedding([query_text])[0]
    q_vec = np.array([q_emb], dtype=np.float32)
    faiss.normalize_L2(q_vec)
    scores, indices = index.search(q_vec, top_k)
    parts = []
    for score, idx in zip(scores[0], indices[0]):
        m = chunks_meta[idx]
        parts.append(
            f"[Std {m['std']} | {m['subj']} | {m['book']} | "
            f"Ch {m['ch_num']}: {m['ch']} | Pages {m['pages']} | "
            f"Relevance: {score:.3f}]\n{m['text']}"
        )
    return "\n---\n".join(parts)

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

app = FastAPI(title="NCERT RAG API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def count_tokens(text: str) -> int:
    return max(1, len(text.encode("utf-8")) // 4)

@app.get("/health")
def health():
    return {"status": "ok", "vectors": index.ntotal, "chunks": len(chunks_meta)}

@app.get("/v1/models")
def list_models():
    return ModelList(data=[ModelInfo(id="ncert-rag")])

@app.post("/v1/chat/completions")
async def chat_completion(req: ChatRequest):
    system_prompt = ""
    conversation = []
    for m in req.messages:
        if m.role == "system":
            system_prompt = m.content
        else:
            conversation.append(f"{m.role}: {m.content}")

    if not conversation:
        raise HTTPException(400, "No messages found")

    question = conversation[-1].replace("user: ", "", 1)
    context = rag(question)

    if not system_prompt:
        system_prompt = (
            "You are a friendly tutor for school children. "
            "Answer based on NCERT textbook excerpts. "
            "Be clear, engaging, and age-appropriate. "
            "For each claim, cite which textbook and chapter it comes from."
        )

    history = "\n".join(conversation[:-1])
    full_prompt = (
        f"{system_prompt}\n\n"
        f"Previous conversation:\n{history}\n\n"
        f"Question: {question}\n\n"
        f"Textbook excerpts:\n{context}"
    )

    p_tokens = count_tokens(full_prompt)

    if req.stream:
        async def stream_generator():
            stream_resp = client.models.generate_content_stream(
                model="gemini-3-flash-preview",
                contents=full_prompt,
            )
            for chunk in stream_resp:
                if chunk.text:
                    sse = {
                        "choices": [{"delta": {"content": chunk.text}, "index": 0}]
                    }
                    yield f"data: {json.dumps(sse)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=full_prompt,
    )
    answer = response.text
    c_tokens = count_tokens(answer)

    return ChatResponse(
        id=f"chatcmpl-{int(time.time())}",
        created=int(time.time()),
        choices=[Choice(message=ChatMessage(role="assistant", content=answer))],
        usage=Usage(
            prompt_tokens=p_tokens,
            completion_tokens=c_tokens,
            total_tokens=p_tokens + c_tokens,
        ),
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
