import json, os, pickle, time, logging, re
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastembed import TextEmbedding

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ncert-rag")

EMBED_DIM = 384
INDEX_URL = os.environ.get("INDEX_URL", "")

log.info("Loading embedding model...")
embed_model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", providers=["CPUExecutionProvider"])
log.info("Model loaded")

index: faiss.Index = None
chunks_meta: list = []

def download_index():
    for name in ["faiss.index", "chunks_meta.pkl"]:
        path = Path(f"/tmp/ncert_rag/{name}")
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
    index = faiss.read_index("/tmp/ncert_rag/faiss.index")
    with open("/tmp/ncert_rag/chunks_meta.pkl", "rb") as f:
        chunks_meta = pickle.load(f)
    log.info(f"Loaded index: {index.ntotal} vectors, {len(chunks_meta)} chunks")

if INDEX_URL:
    Path("/tmp/ncert_rag").mkdir(exist_ok=True)
    download_index()
load_index()

def get_embedding(texts: list[str]) -> list[list[float]]:
    emb_gen = embed_model.embed([t[:2000] for t in texts])
    return [list(e) for e in emb_gen]

# --- API Models ---
class RAGQuery(BaseModel):
    query: str
    top_k: int = 5

class ChapterQuery(BaseModel):
    chapter: str
    std: Optional[str] = None
    subject: Optional[str] = None

class SourceInfo(BaseModel):
    std: str
    subject: str
    textbook: str
    chapter: str
    chapter_number: str
    pages: list[int]

class RAGResult(BaseModel):
    text: str
    source: SourceInfo
    relevance_score: float
    image_count: int

class RAGResponse(BaseModel):
    object: str = "rag.query"
    query: str
    results: list[RAGResult]
    total_results: int

class ChapterChunk(BaseModel):
    text: str
    pages: list[int]
    image_count: int

class ChapterResult(BaseModel):
    std: str
    subject: str
    textbook: str
    chapter: str
    chapter_number: str
    total_chunks: int
    chunks: list[ChapterChunk]

class ChapterResponse(BaseModel):
    object: str = "rag.chapter"
    query: str
    matches: list[ChapterResult]
    total_matches: int

class HealthResponse(BaseModel):
    status: str
    vectors: int
    chunks: int

app = FastAPI(title="NCERT RAG Tool")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
def health() -> HealthResponse:
    return HealthResponse(status="ok", vectors=index.ntotal, chunks=len(chunks_meta))

@app.post("/v1/rag/query")
def rag_query(req: RAGQuery) -> RAGResponse:
    q_emb = get_embedding([req.query])[0]
    q_vec = np.array([q_emb], dtype=np.float32)
    scores, indices = index.search(q_vec, req.top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        m = chunks_meta[idx]
        results.append(RAGResult(
            text=m["text"],
            source=SourceInfo(
                std=m["std"],
                subject=m["subj"],
                textbook=m["book"],
                chapter=m["ch"],
                chapter_number=m["ch_num"],
                pages=m.get("pages", []),
            ),
            relevance_score=round(float(score), 4),
            image_count=len(m.get("images", [])),
        ))
    return RAGResponse(query=req.query, results=results, total_results=len(results))

@app.post("/v1/rag/chapter")
def get_chapter(req: ChapterQuery) -> ChapterResponse:
    name_lower = req.chapter.strip().lower()
    pattern = re.escape(name_lower)
    matched_chunks = []
    for m in chunks_meta:
        ch_lower = m["ch"].strip().lower()
        if not re.search(pattern, ch_lower):
            continue
        if req.std and m["std"] != req.std:
            continue
        if req.subject and m["subj"].lower() != req.subject.lower():
            continue
        matched_chunks.append(m)
    if not matched_chunks:
        raise HTTPException(404, f"No chapters matching '{req.chapter}'")
    groups = {}
    for m in matched_chunks:
        key = (m["std"], m["subj"], m["book"], m["ch"], m["ch_num"])
        if key not in groups:
            groups[key] = []
        groups[key].append(m)
    matches = []
    for (std, subj, book, ch, ch_num), chunks in groups.items():
        matches.append(ChapterResult(
            std=std, subject=subj, textbook=book,
            chapter=ch, chapter_number=ch_num,
            total_chunks=len(chunks),
            chunks=[ChapterChunk(
                text=c["text"], pages=c.get("pages", []),
                image_count=len(c.get("images", []))
            ) for c in chunks],
        ))
    return ChapterResponse(query=req.chapter, matches=matches, total_matches=len(matches))

@app.get("/v1/rag/chapters")
def list_chapters():
    seen = set()
    chapters = []
    for m in chunks_meta:
        key = (m["std"], m["subj"], m["book"], m["ch"])
        if key not in seen:
            seen.add(key)
            chapters.append({
                "std": m["std"],
                "subject": m["subj"],
                "textbook": m["book"],
                "chapter": m["ch"],
                "chapter_number": m["ch_num"],
            })
    return {"object": "rag.chapters", "chapters": chapters, "total": len(chapters)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
