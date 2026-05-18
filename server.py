import json, os, pickle, time, logging, re, traceback, tarfile, io, threading
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, field_validator
from fastembed import TextEmbedding

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ncert-rag")

EMBED_DIM = 384
INDEX_URL = os.environ.get("INDEX_URL", "")
IMAGES_URL = os.environ.get("IMAGES_URL", INDEX_URL)
CACHE_DIR = Path("/tmp/ncert_rag")
CLASS_ARCHIVES_DIR = CACHE_DIR / "archives"

index: faiss.Index = None
chunks_meta: list = []
prefix_class_map: dict = {}
class_archives_lock = threading.Lock()
downloading_classes = set()
extraction_semaphore = threading.Semaphore(1)

def download_file(name: str, url_base: str, dest: Path):
    url = f"{url_base}/{name}"
    log.info(f"Downloading {url}...")
    r = requests.get(url, timeout=300, stream=True)
    r.raise_for_status()
    size = 0
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                size += len(chunk)
    log.info(f"Downloaded {name} ({size//1024} KB)")

def download_index():
    for name in ["faiss.index", "chunks_meta.pkl"]:
        path = CACHE_DIR / name
        if path.exists():
            log.info(f"Using cached {name}")
            continue
        download_file(name, INDEX_URL, path)

def load_index():
    global index, chunks_meta, prefix_class_map
    index = faiss.read_index(str(CACHE_DIR / "faiss.index"))
    with open(CACHE_DIR / "chunks_meta.pkl", "rb") as f:
        chunks_meta = pickle.load(f)
    for m in chunks_meta:
        std = m["std"]
        for img in m.get("images", []):
            prefix = img.split("_")[0]
            if prefix not in prefix_class_map:
                prefix_class_map[prefix] = std
    log.info(f"Loaded index: {index.ntotal} vectors, {len(chunks_meta)} chunks")
    log.info(f"Mapped {len(prefix_class_map)} image prefixes to classes")

def get_class_extract_dir(std: str) -> Path:
    return CLASS_ARCHIVES_DIR / std

def get_class_archive_path(std: str) -> Path:
    return CLASS_ARCHIVES_DIR / f"class_{std}.tar.gz"

def download_class_archive(std: str):
    with extraction_semaphore:
        with class_archives_lock:
            if get_class_extract_dir(std).exists():
                downloading_classes.discard(std)
                return
        tar_path = get_class_archive_path(std)
        extract_dir = get_class_extract_dir(std)
        try:
            CLASS_ARCHIVES_DIR.mkdir(parents=True, exist_ok=True)
            download_file(f"class_{std}.tar.gz", IMAGES_URL, tar_path)
            log.info(f"Extracting class {std} archive...")
            extract_dir.mkdir(exist_ok=True)
            with tarfile.open(str(tar_path), "r:gz") as tar:
                tar.extractall(path=str(extract_dir))
            tar_path.unlink()
            count = len(list(extract_dir.iterdir()))
            log.info(f"Class {std} images ready: {count} files in {extract_dir}")
        except Exception as e:
            log.warning(f"Failed to download/extract class {std} archive: {e}")
            if tar_path.exists():
                tar_path.unlink()
        finally:
            with class_archives_lock:
                downloading_classes.discard(std)

def ensure_class_archive(std: str):
    extract_dir = get_class_extract_dir(std)
    if extract_dir.exists():
        return True
    with class_archives_lock:
        if extract_dir.exists():
            return True
        if std in downloading_classes:
            return False
        downloading_classes.add(std)
    threading.Thread(target=download_class_archive, args=(std,), daemon=True).start()
    return False

def serve_image_from_class_archive(filename: str) -> Optional[bytes]:
    prefix = filename.split("_")[0]
    std = prefix_class_map.get(prefix)
    if not std:
        log.warning(f"No class mapping for prefix '{prefix}' in '{filename}'")
        return None
    extract_dir = get_class_extract_dir(std)
    if not extract_dir.exists():
        ready = ensure_class_archive(std)
        if not ready:
            return None
    if not extract_dir.exists():
        return None
    img_path = extract_dir / filename
    if not img_path.exists():
        return None
    try:
        return img_path.read_bytes()
    except Exception as e:
        log.warning(f"Error reading {filename}: {e}")
        return None

log.info("Loading embedding model...")
embed_model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", providers=["CPUExecutionProvider"])
log.info("Model loaded")

if INDEX_URL:
    CACHE_DIR.mkdir(exist_ok=True)
    download_index()
load_index()

def get_embedding(texts: list[str]) -> list[list[float]]:
    emb_gen = embed_model.embed([t[:2000] for t in texts])
    return [list(e) for e in emb_gen]

# --- API Models ---
class RAGQuery(BaseModel):
    query: str
    top_k: int = 5

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v):
        if not v.strip():
            raise ValueError("query must not be empty")
        return v

    @field_validator("top_k")
    @classmethod
    def top_k_positive(cls, v):
        if v < 1:
            raise ValueError("top_k must be >= 1")
        return v

class ChapterQuery(BaseModel):
    chapter: str
    std: Optional[str] = None
    subject: Optional[str] = None

    @field_validator("chapter")
    @classmethod
    def chapter_not_empty(cls, v):
        if not v.strip():
            raise ValueError("chapter must not be empty")
        return v

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
    images_available: int = 0

app = FastAPI(title="NCERT RAG Tool")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    log.error(f"Unhandled exception on {request.method} {request.url.path}: {exc}")
    log.error(traceback.format_exc())
    return JSONResponse(status_code=500, content={"detail": str(exc), "type": type(exc).__name__})

def build_image_urls(images: list[str]) -> list[str]:
    return [f"/v1/rag/image/{img.replace('.png', '.jpg')}" for img in images]

@app.get("/health")
def health() -> HealthResponse:
    cached = len([p for p in CLASS_ARCHIVES_DIR.iterdir() if p.is_dir()]) if CLASS_ARCHIVES_DIR.exists() else 0
    return HealthResponse(status="ok", vectors=index.ntotal, chunks=len(chunks_meta), images_available=cached)

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
    prefix = safe.split("_")[0]
    std = prefix_class_map.get(prefix)
    if not std:
        raise HTTPException(404, "Image not found")
    extract_dir = get_class_extract_dir(std)
    if not extract_dir.exists():
        ready = ensure_class_archive(std)
        if not ready:
            raise HTTPException(503, detail=f"Class {std} images loading, retry in 30s")
    data = serve_image_from_class_archive(safe)
    if data is None:
        raise HTTPException(404, "Image not found in class {std}")
    return Response(content=data, media_type="image/jpeg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
