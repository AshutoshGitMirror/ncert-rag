import json, os, pickle, time, logging, re, traceback, tarfile, io
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from fastembed import TextEmbedding

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ncert-rag")

EMBED_DIM = 384
INDEX_URL = os.environ.get("INDEX_URL", "")
IMAGES_URL = os.environ.get("IMAGES_URL", "")
CACHE_DIR = Path("/tmp/ncert_rag")
IMAGES_DIR = CACHE_DIR / "images"
IMAGES_TAR = CACHE_DIR / "images.tar.gz"

log.info("Loading embedding model...")
embed_model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", providers=["CPUExecutionProvider"])
log.info("Model loaded")

index: faiss.Index = None
chunks_meta: list = []
images_ready = False

def download_file(name: str, url_base: str, dest: Path):
    url = f"{url_base}/{name}"
    log.info(f"Downloading {url}...")
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    with open(dest, "wb") as f:
        f.write(r.content)
    log.info(f"Downloaded {name} ({len(r.content)//1024} KB)")

def download_index():
    for name in ["faiss.index", "chunks_meta.pkl"]:
        path = CACHE_DIR / name
        if path.exists():
            log.info(f"Using cached {name}")
            continue
        download_file(name, INDEX_URL, path)

def load_index():
    global index, chunks_meta
    index = faiss.read_index(str(CACHE_DIR / "faiss.index"))
    with open(CACHE_DIR / "chunks_meta.pkl", "rb") as f:
        chunks_meta = pickle.load(f)
    log.info(f"Loaded index: {index.ntotal} vectors, {len(chunks_meta)} chunks")

def download_images():
    global images_ready
    if not IMAGES_URL:
        log.info("No IMAGES_URL set, skipping image download")
        return
    if IMAGES_TAR.exists():
        log.info(f"Using cached images.tar.gz ({IMAGES_TAR.stat().st_size // 1024 // 1024} MB)")
        images_ready = True
        return
    try:
        download_file("images.tar.gz", IMAGES_URL, IMAGES_TAR)
        images_ready = True
    except Exception as e:
        log.warning(f"Failed to download images: {e}")

def serve_image_from_tar(filename: str) -> Optional[bytes]:
    if not IMAGES_TAR.exists():
        return None
    try:
        with tarfile.open(str(IMAGES_TAR), "r:gz") as tar:
            member = tar.extractfile(filename)
            if member:
                return member.read()
        return None
    except Exception as e:
        log.warning(f"Error reading {filename} from archive: {e}")
        return None

if INDEX_URL:
    CACHE_DIR.mkdir(exist_ok=True)
    download_index()
load_index()
download_images()

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
    image_urls: list[str] = []

class RAGResponse(BaseModel):
    object: str = "rag.query"
    query: str
    results: list[RAGResult]
    total_results: int

class ChapterChunk(BaseModel):
    text: str
    pages: list[int]
    image_count: int
    image_urls: list[str] = []

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
    images_ready: bool = False

app = FastAPI(title="NCERT RAG Tool")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    log.error(f"Unhandled exception on {request.method} {request.url.path}: {exc}")
    log.error(traceback.format_exc())
    return JSONResponse(status_code=500, content={"detail": str(exc), "type": type(exc).__name__})

def build_image_urls(images: list[str]) -> list[str]:
    if not images_ready:
        return []
    return [f"/v1/rag/image/{img.replace('.png', '.jpg')}" for img in images]

@app.get("/health")
def health() -> HealthResponse:
    return HealthResponse(status="ok", vectors=index.ntotal, chunks=len(chunks_meta), images_ready=images_ready)

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
                chapter_number=str(m["ch_num"]),
                pages=m.get("pages", []),
            ),
            relevance_score=round(float(score), 4),
            image_count=len(m.get("images", [])),
            image_urls=build_image_urls(m.get("images", [])),
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
            chapter=ch, chapter_number=str(ch_num),
            total_chunks=len(chunks),
            chunks=[ChapterChunk(
                text=c["text"], pages=c.get("pages", []),
                image_count=len(c.get("images", [])),
                image_urls=build_image_urls(c.get("images", [])),
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

@app.get("/v1/rag/image/{filename:path}")
def get_image(filename: str):
    safe = Path(filename).name
    data = serve_image_from_tar(safe)
    if data is None:
        raise HTTPException(404, "Image not found")
    return Response(content=data, media_type="image/jpeg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
