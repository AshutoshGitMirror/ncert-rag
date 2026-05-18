# NCERT RAG API

Semantic search over every NCERT textbook (Classes 1–12, all subjects). 19,678 indexed chunks across 1,089 chapters, with 79,617 textbook images served on demand. Deployed as a zero-external-dependency API — no API keys, no external LLM calls, all embedding is local.

**Base URL**: `https://ncert-rag-api.onrender.com`

---

## Table of Contents

- [Motivation](#motivation)
- [Architecture](#architecture)
- [Quickstart](#quickstart)
- [API Reference](#api-reference)
  - [Health](#get-health)
  - [List Chapters](#get-v1ragchapters)
  - [Semantic Query](#post-v1ragquery)
  - [Chapter Lookup](#post-v1ragchapter)
  - [Image Serving](#get-v1ragimagefilename)
  - [Errors](#error-responses)
- [Usage Examples](#usage-examples)
  - [cURL](#curl)
  - [Python](#python)
  - [Node.js / TypeScript](#nodejs--typescript)
  - [LangChain / LlamaIndex](#langchain--llamaindex)
- [Data Pipeline](#data-pipeline)
- [Image Processing](#image-processing)
- [Deployment Guide](#deployment-guide)
  - [render.yaml](#renderyaml)
  - [Environment Variables](#environment-variables)
  - [Manual Deploy](#manual-deploy)
  - [Cold Start Behaviour](#cold-start-behaviour)
- [Development](#development)
  - [Local Setup](#local-setup)
  - [Rebuilding the Index](#rebuilding-the-index)
- [Dataset Statistics](#dataset-statistics)
- [Constraints & Limitations](#constraints--limitations)
- [Troubleshooting](#troubleshooting)
- [Repository Contents](#repository-contents)

---

## Motivation

NCERT textbooks are the de-facto standard for K-12 education across India. They cover every subject — Mathematics, Physics, Chemistry, Biology, History, Geography, Economics, Political Science, Fine Arts, Biotechnology, and more — from Class 1 through Class 12.

This API turns that entire corpus into a **callable vector search tool** for any AI agent. Instead of building RAG from scratch, any application can:

- Retrieve relevant textbook passages by **semantic meaning** (not just keyword match)
- Look up **full chapter content** by name with optional class/subject filters
- Access **diagrams and illustrations** alongside the text

There are no API keys, no usage limits, and no external dependencies at runtime.

---

## Architecture

```
┌──────────────┐     ┌──────────────────────┐     ┌──────────────────┐
│  AI Agent    │────▶│  FastAPI (Uvicorn)    │────▶│  FAISS IndexFlatIP│
│  (curl/Python│     │  Render Free Tier     │     │  19,678 × 384-dim │
│   /LangChain)│     │  Python 3.12-slim     │     │  Cosine Similarity │
└──────────────┘     │  fastembed ONNX CPU   │     └──────────────────┘
                     │  512 MB RAM, 0.1 CPU  │              │
                     └───────────┬──────────┘              │
                                 │                         │
                          ┌──────┴──────┐          ┌───────┴────────┐
                          │  GitHub     │          │  /tmp/ncert_rag │
                          │  Releases   │          │  (ephemeral     │
                          │  (v1):      │          │   cache on      │
                          │  · faiss.index         │   disk)         │
                          │  · chunks_meta.pkl     │                 │
                          │  · class_*.tar.gz (12) │  Images cached  │
                          └─────────────┘          │  by class dir   │
                                                   └────────────────┘
```

### Component details

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Server** | FastAPI + Uvicorn | HTTP API, request handling, CORS |
| **Embedding** | fastembed (`sentence-transformers/all-MiniLM-L6-v2`) | Convert text to 384-dim vectors via ONNX runtime |
| **Vector Index** | FAISS `IndexFlatIP` | Brute-force inner product search (cosine sim on normalized vectors) |
| **Text Storage** | Pickle (`chunks_meta.pkl`) | Full chunk text, source metadata, image references |
| **Image Storage** | Per-class tar.gz archives on GitHub Releases | 12 archives (35–133 MB each), lazy-downloaded and extracted |
| **Hosting** | Render Free Web Service (Docker) | 512 MB RAM, 0.1 CPU, Singapore region |
| **CDN** | GitHub Releases → CloudFront/S3 | Asset delivery for index + images |

### Key design decisions

- **No external AI dependencies**: All embedding runs locally via ONNX. No API keys, no rate limits, no cost per query.
- **No persistent disk**: Render free tier doesn't support persistent disks. The server downloads + caches to `/tmp` on cold start, which is ephemeral but survives within a container's lifetime.
- **Lazy image loading**: Images are not downloaded at server startup. Each class archive is fetched on first request, then extracted to disk. Subsequent requests for that class are instant.
- **All-MiniLM-L6-v2**: 384-dimensional embeddings. Chosen for speed (fast even on CPU) and competitive retrieval quality. Dimensionality reduction from 768→384 had negligible impact on textbook retrieval.

---

## Quickstart

```bash
# Health check
curl https://ncert-rag-api.onrender.com/health

# Search for "photosynthesis"
curl -X POST https://ncert-rag-api.onrender.com/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query": "photosynthesis", "top_k": 2}'

# Look up chapter "Laws of Motion" in Class 9 Science
curl -X POST https://ncert-rag-api.onrender.com/v1/rag/chapter \
  -H "Content-Type: application/json" \
  -d '{"chapter": "Laws of Motion", "std": "9", "subject": "Science"}'

# Fetch an image from a result
curl -O https://ncert-rag-api.onrender.com/v1/rag/image/kebo111_p21_0.jpg
```

---

## API Reference

### `GET /health`

Server health and index statistics.

**Response** (HTTP 200):
```json
{
  "status": "ok",
  "vectors": 19678,
  "chunks": 19678,
  "images_available": 3
}
```

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | `"ok"` if the server is operational |
| `vectors` | int | Total vectors in the FAISS index |
| `chunks` | int | Total text chunks indexed |
| `images_available` | int | Number of class-level image archives currently cached |

---

### `GET /v1/rag/chapters`

List every unique chapter across all textbooks (deduplicated by standard + subject + textbook + chapter name).

**Response** (HTTP 200):
```json
{
  "object": "rag.chapters",
  "total": 1089,
  "chapters": [
    {
      "std": "1",
      "subject": "Mathematics",
      "textbook": "Joyful Mathematics-1",
      "chapter": "Finding the Furry Cat! (Pre-number Concepts)",
      "chapter_number": "1"
    }
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `total` | int | Total unique chapters |
| `chapters[].std` | string | Class level ("1"–"12") |
| `chapters[].subject` | string | Subject name |
| `chapters[].textbook` | string | Textbook title |
| `chapters[].chapter` | string | Chapter title |
| `chapters[].chapter_number` | string | Chapter number |

---

### `POST /v1/rag/query`

Semantic search over all indexed textbook chunks. Uses cosine similarity (FAISS IndexFlatIP on normalized vectors) to find the most relevant passages for a free-text query.

**Request**:
```json
{
  "query": "photosynthesis",
  "top_k": 5
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | **required** | Natural language query (max 2000 chars per chunk) |
| `top_k` | int | 5 | Number of results to return (max 19678) |

**Response** (HTTP 200):
```json
{
  "object": "rag.query",
  "query": "photosynthesis",
  "total_results": 2,
  "results": [
    {
      "text": "PHOTOSYNTHESIS IN HIGHER PLANTS\n151\nSUMMARY\nGreen plants make their own food by photosynthesis...",
      "source": {
        "std": "11",
        "subject": "Biology",
        "textbook": "Biology",
        "chapter": "Photosynthesis in Higher Plants",
        "chapter_number": "11",
        "pages": [21]
      },
      "relevance_score": 0.7623,
      "image_count": 2,
      "image_urls": [
        "/v1/rag/image/kebo111_p21_0.jpg",
        "/v1/rag/image/kebo111_p21_1.jpg"
      ]
    }
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `results[].text` | string | Retrieved passage (~500 tokens) |
| `results[].source.std` | string | Class level |
| `results[].source.subject` | string | Subject name |
| `results[].source.textbook` | string | Textbook title |
| `results[].source.chapter` | string | Chapter title |
| `results[].source.chapter_number` | string | Chapter number |
| `results[].source.pages` | list[int] | Page numbers in the source PDF |
| `results[].relevance_score` | float | Cosine similarity (0–1). Higher = more relevant |
| `results[].image_count` | int | Number of images embedded in this chunk |
| `results[].image_urls` | list[str] | Relative URLs to fetch images |

**Notes**:
- Results are sorted by `relevance_score` descending.
- `image_urls` are relative paths (e.g. `/v1/rag/image/...`). Prepend the base URL to fetch.
- The first time an image from a new class is requested, the server downloads and extracts the archive (takes 30–60s). Subsequent requests are instant.
- `relevance_score` for queries with no semantically close match may return low scores (0.3–0.4) with tangentially related content.

---

### `POST /v1/rag/chapter`

Retrieve all chunk content for a chapter by name. Supports partial, case-insensitive matching on the chapter title, with optional filters for class (`std`) and `subject`.

**Request**:
```json
{
  "chapter": "matter",
  "std": "11",
  "subject": "Physics"
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `chapter` | string | **required** | Chapter name substring (case-insensitive, regex-safe) |
| `std` | string | `null` | Filter by class ("1"–"12") |
| `subject` | string | `null` | Filter by subject (exact, case-insensitive) |

**Response** (HTTP 200):
```json
{
  "object": "rag.chapter",
  "query": "matter",
  "total_matches": 1,
  "matches": [
    {
      "std": "11",
      "subject": "Physics",
      "textbook": "Physics Part-II",
      "chapter": "Thermal Properties of Matter",
      "chapter_number": "3",
      "total_chunks": 24,
      "chunks": [
        {
          "text": "CHAPTER TEN\nTHERMAL PROPERTIES OF MATTER\n10.1 INTRODUCTION...",
          "pages": [1],
          "image_count": 8,
          "image_urls": [
            "/v1/rag/image/gegp108_p20_0.jpg"
          ]
        }
      ]
    }
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `matches[].std` | string | Class level |
| `matches[].subject` | string | Subject name |
| `matches[].textbook` | string | Textbook title |
| `matches[].chapter` | string | Chapter title |
| `matches[].chapter_number` | string | Chapter number |
| `matches[].total_chunks` | int | Total chunks for this chapter |
| `matches[].chunks[].text` | string | Chunk text content |
| `matches[].chunks[].pages` | list[int] | Page numbers |
| `matches[].chunks[].image_count` | int | Number of images in this chunk |
| `matches[].chunks[].image_urls` | list[str] | Relative URLs to fetch images |

**Filtering examples**:

| Request | Effect |
|---------|--------|
| `{"chapter": "matter"}` | All chapters matching "matter" across all classes and subjects |
| `{"chapter": "matter", "std": "11"}` | Only Class 11 chapters |
| `{"chapter": "matter", "subject": "Physics"}` | Only Physics chapters |
| `{"chapter": "matter", "std": "11", "subject": "Physics"}` | Class 11 Physics chapters only |

**Error response** (HTTP 404):
```json
{"detail": "No chapters matching 'xyznonexistent'"}
```

---

### `GET /v1/rag/image/{filename}`

Serve a textbook image by its filename. Images are extracted from PDFs, converted from PNG to JPEG (quality 85), and served on demand from a class-level archive.

**Example**:
```bash
curl https://ncert-rag-api.onrender.com/v1/rag/image/kebo111_p21_0.jpg -o diagram.jpg
```

**Behaviour**:
1. **First request for a class**: Returns HTTP 503 with `{"detail": "Class N images loading, retry in 30s"}`. The server starts downloading that class's image archive in the background.
2. **Subsequent requests** (after ~30–60s): Returns HTTP 200 with `image/jpeg` content.
3. **Cache hit** (archive already extracted): Returns HTTP 200 instantly.

| Response | Condition |
|----------|-----------|
| `200 OK` | Image served as `image/jpeg` |
| `404 Not Found` | Image filename not in the archive |
| `503 Service Unavailable` | Class archive is being downloaded; retry after 30s |

**Image naming convention**: Images follow the pattern `{prefix}_{page}_{index}.jpg` where:
- `prefix`: textbook identifier (e.g. `kebo` = Class 11 Biology)
- `page`: page number in the PDF
- `index`: image index on that page (0-based)

---

### Error Responses

All endpoints return structured errors consistent with FastAPI conventions:

**Validation error** (HTTP 422):
```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "query"],
      "msg": "Field required"
    }
  ]
}
```

**Not found** (HTTP 404):
```json
{"detail": "No chapters matching 'xyznonexistent'"}
```

**Server error** (HTTP 500):
```json
{
  "detail": "Error message",
  "type": "ExceptionType"
}
```

---

## Usage Examples

### cURL

```bash
BASE="https://ncert-rag-api.onrender.com"

# Health
curl $BASE/health | jq

# Count chapters
curl $BASE/v1/rag/chapters | jq '.total'

# Semantic search
curl -X POST $BASE/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query": "newton laws of motion", "top_k": 3}' | jq '.results[].source'

# Chapter lookup with filters
curl -X POST $BASE/v1/rag/chapter \
  -H "Content-Type: application/json" \
  -d '{"chapter": "force", "std": "9", "subject": "Science"}' | jq '.matches[].chapter'

# Download first image from a query result
IMG=$(curl -s -X POST $BASE/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query": "photosynthesis"}' | jq -r '.results[0].image_urls[0]')
curl -o photosynthesis.jpg "$BASE$IMG"
```

### Python

```python
import requests

BASE = "https://ncert-rag-api.onrender.com"

def search(query: str, top_k: int = 5):
    """Search NCERT textbooks by semantic meaning."""
    resp = requests.post(f"{BASE}/v1/rag/query", json={"query": query, "top_k": top_k})
    resp.raise_for_status()
    return resp.json()

def get_chapter(chapter: str, std: str = None, subject: str = None):
    """Get full chapter content by name."""
    body = {"chapter": chapter}
    if std: body["std"] = std
    if subject: body["subject"] = subject
    resp = requests.post(f"{BASE}/v1/rag/chapter", json=body)
    resp.raise_for_status()
    return resp.json()

def fetch_image(url: str) -> bytes:
    """Download an image by its relative URL."""
    resp = requests.get(f"{BASE}{url}")
    resp.raise_for_status()
    return resp.content

# Example: find and fetch images for "photosynthesis"
results = search("photosynthesis", top_k=2)
for r in results["results"]:
    s = r["source"]
    print(f"[{s['std']} {s['subject']}] {s['chapter']} (score={r['relevance_score']})")
    for img_url in r["image_urls"]:
        img_data = fetch_image(img_url)
        filename = img_url.split("/")[-1]
        with open(filename, "wb") as f:
            f.write(img_data)
        print(f"  Saved {filename} ({len(img_data)} bytes)")

# Example: get full chapter content
chapter = get_chapter("matter", std="11", subject="Physics")
for match in chapter["matches"]:
    print(f"\n{match['chapter']} ({match['total_chunks']} chunks)")
    for chunk in match["chunks"][:2]:  # first 2 chunks
        print(f"  Pages {chunk['pages']}: {chunk['text'][:80]}...")
```

### Node.js / TypeScript

```typescript
const BASE = "https://ncert-rag-api.onrender.com";

async function search(query: string, topK = 5) {
  const res = await fetch(`${BASE}/v1/rag/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, top_k: topK }),
  });
  return res.json();
}

// Find and download images
const data = await search("photosynthesis");
for (const result of data.results) {
  for (const imgUrl of result.image_urls) {
    const res = await fetch(`${BASE}${imgUrl}`);
    const buffer = await res.arrayBuffer();
    console.log(`Downloaded ${imgUrl} (${buffer.byteLength} bytes)`);
  }
}
```

### LangChain / LlamaIndex

The API is compatible with any tool-calling agent. Example function definition for an LLM:

```json
{
  "type": "function",
  "function": {
    "name": "ncert_rag_query",
    "description": "Search NCERT textbooks for relevant passages by semantic meaning",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {
          "type": "string",
          "description": "Natural language query about any NCERT topic"
        },
        "top_k": {
          "type": "integer",
          "description": "Number of results to return (1-20)",
          "default": 5
        }
      }
    }
  }
}
```

---

## Data Pipeline

The index was built in 4 phases on a Google Colab VM (CPU-only, ~45 minutes total):

### Phase 1: PDF Download

- Source: NCERT official website (`https://ncert.nic.in/textbook/...`)
- 1,091 PDF textbooks across classes 1–12
- Handled 5 dead URLs gracefully (created empty marker files)
- Total: ~3.2 GB of PDFs

### Phase 2: Text & Image Extraction

- Library: PyMuPDF (`fitz`)
- Per page: extract text content + all embedded images as PNG
- Output: 82,124 PNG images extracted alongside text
- Images are diagrams, graphs, maps, illustrations — NOT photographs

### Phase 3: Chunking

- Strategy: Section-based splitting (chapter headings as boundaries)
- Target: ~500 tokens per chunk (~2000 characters)
- Overlap: None (sections are naturally disjoint)
- Result: 19,678 chunks from 1,089 chapters

### Phase 4: Embedding & Indexing

- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Dimension: 384
- Framework: sentence-transformers (PyTorch) on Colab CPU
- Index: FAISS `IndexFlatIP` (brute-force inner product)
- Normalization: Embeddings are L2-normalized, so inner product = cosine similarity
- Duration: ~45 minutes for 19,678 texts

### Post-processing (this session)

- Converted all 82,124 PNG images to JPEG (quality 85, optimize=True) → 1.97 GB
- Compressed into 12 per-class tar.gz archives (35–133 MB each)
- Uploaded to GitHub Releases v1 alongside the index

---

## Image Processing

### Why JPEG conversion?

The original PDF extraction produced PNG images (lossless, large). PNG was used during pipeline development to preserve quality for debugging. For production serving:

| Format | Size (total) | Avg per image |
|--------|-------------|---------------|
| PNG (original) | 7.0 GB | 89 KB |
| JPEG Q85 (converted) | 1.97 GB | 25 KB |
| tar.gz (compressed) | 0.99 GB per full archive | — |

JPEG quality 85 provides visually lossless results for textbook content (diagrams, graphs, text screenshots) at ~25% of the original size.

### Image naming

Images are named `{textbook_code}{p|_}{page}_{index}.png` originally (e.g., `kebo111_p21_0.png`). The server maps `.png` → `.jpg` when serving. The `textbook_code` prefix maps each image to its class level via the chunk metadata.

### Supported image types

The server serves `image/jpeg`. If an image was extracted as PNG originally, it is converted to JPEG during archive preparation.

---

## Deployment Guide

The API is designed to run on Render's free tier. It can also be deployed to any Docker-compatible platform.

### render.yaml

```yaml
services:
  - type: web
    name: ncert-rag-api
    runtime: docker
    repo: https://github.com/AshutoshGitMirror/ncert-rag
    plan: free
    region: singapore
    envVars:
      - key: INDEX_URL
        value: https://github.com/AshutoshGitMirror/ncert-rag/releases/download/v1
      - key: IMAGES_URL
        value: https://github.com/AshutoshGitMirror/ncert-rag/releases/download/v1
      - key: PORT
        value: "8000"
    healthCheckPath: /health
```

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `INDEX_URL` | Yes | — | Base URL for `faiss.index` and `chunks_meta.pkl` |
| `IMAGES_URL` | No | `INDEX_URL` | Base URL for `class_N.tar.gz` archives |
| `PORT` | No | `8000` | Server port |

### Manual Deploy

```bash
curl -X POST https://api.render.com/v1/services/<SERVICE_ID>/deploys \
  -H "Authorization: Bearer $RENDER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"clearCache": "clear"}'
```

### Cold Start Behaviour

When the server starts (or restarts after idle spin-down):

1. **Model loading** (~5s): Downloads and loads the ONNX model for `all-MiniLM-L6-v2` via fastembed
2. **Index loading** (~10s): Downloads `faiss.index` (28.8 MB) + `chunks_meta.pkl` (34.8 MB) from GitHub Releases
3. **Image archives**: NOT downloaded at startup. Each class archive is lazily fetched on first image request.

Text queries work immediately after startup. Images are available per-class after the first request triggers a 30–60s download + extraction.

---

## Development

### Local Setup

```bash
# Clone
git clone https://github.com/AshutoshGitMirror/ncert-rag.git
cd ncert-rag

# Install dependencies
pip install -r requirements.txt

# Set env vars
export INDEX_URL="https://github.com/AshutoshGitMirror/ncert-rag/releases/download/v1"
export IMAGES_URL="$INDEX_URL"

# Run
uvicorn server:app --host 0.0.0.0 --port 8000
```

### Rebuilding the Index

The index was built on Google Colab. To rebuild:

1. Run the pipeline notebook (`ncert_rag.ipynb`):
   - Phase 1: Download 1,091 PDFs
   - Phase 2: Extract text + images
   - Phase 3: Chunk by section
   - Phase 4: Embed with all-MiniLM-L6-v2 → FAISS

2. Upload artifacts to GitHub:
   ```bash
   # Upload new faiss.index + chunks_meta.pkl
   gh release upload v1 faiss.index chunks_meta.pkl --clobber

   # Create and upload image archives
   python3 -c "
   import tarfile, os, pickle
   # ... (see class archive creation in pipeline)
   "
   ```

---

## Dataset Statistics

| Stat | Value |
|------|-------|
| PDFs processed | 1,091 |
| Total chunks | 19,678 |
| Total chapters | 1,089 |
| Unique subjects | 27 |
| Classes covered | 1–12 |
| Embedding model | all-MiniLM-L6-v2 |
| Embedding dimension | 384 |
| Index type | FAISS IndexFlatIP |
| Images extracted | 82,124 (PNG) |
| Images served | 79,617 (JPEG) |
| Image archives | 12 (per-class tar.gz) |
| Archive size range | 35–133 MB |

### Per-class image distribution

| Class | Images | Archive Size |
|-------|--------|-------------|
| 1 | 2,216 | 52 MB |
| 2 | 1,588 | 40 MB |
| 3 | 6,786 | 89 MB |
| 4 | 4,218 | 77 MB |
| 5 | 3,904 | 61 MB |
| 6 | 5,720 | 71 MB |
| 7 | 9,567 | 110 MB |
| 8 | 7,110 | 108 MB |
| 9 | 6,756 | 133 MB |
| 10 | 3,799 | 35 MB |
| 11 | 14,023 | 122 MB |
| 12 | 13,930 | 119 MB |

### Available subjects

Accountancy, Arts, Biology, Biotechnology, Business Studies, Chemistry, Computer Science, Economics, English, Fine Art, Geography, Health and Physical Education, History, Home Science, Informatics Practices, Knowledge Traditions Practices of India, Mathematics, Physical Education and Well Being, Physics, Political Science, Psychology, Science, Skill Education, Social Science, Sociology, The World Around Us, Vocational Education

---

## Constraints & Limitations

### Runtime constraints (Render Free Tier)

| Constraint | Value | Impact |
|-----------|-------|--------|
| RAM | 512 MB | Cannot keep all 12 image archives extracted simultaneously (max ~3–4 classes before OOM) |
| CPU | 0.1 vCPU | ~30–60s per image archive download + extraction |
| Idle timeout | 15 minutes | Server spins down; cold start for the next request |
| Bandwidth | 100 GB/month | Sufficient for moderate usage |
| Ephemeral storage | Shared node disk | `/tmp` is not RAM-limited; can hold multiple extracted image archives |

### Image serving behaviour

- **First request for a new class**: Returns HTTP 503 while archive downloads (~30–60s)
- **Multiple classes**: Each class is independently cached. A query spanning classes 6, 11, and 12 would need 3 separate downloads
- **Cache persistence**: Archives persist across requests within a container's lifetime. On idle spin-down (15 min), the container is destroyed and caches are lost

### Embedding limitations

- `all-MiniLM-L6-v2` is a general-purpose embedding model. It may not capture highly domain-specific terminology as well as a fine-tuned model would
- Max input length: 256 WordPiece tokens (~2000 characters). Longer texts are truncated
- The model is English-only. Hindi and other Indian language NCERT texts are not indexed

### Data quality notes

- Chunking is section-based (chapter headings). Some very large sections were not further subdivided
- Image extraction from PDFs may miss some embedded SVGs or vector graphics. Only raster images (JPEG2000, PNG) embedded in the PDF are extracted
- ~2,500 zero-byte images (corrupted entries in source PDFs) were excluded

---

## Troubleshooting

### "Class X images loading, retry in 30s"

The archive for that class hasn't been downloaded yet. Wait 30–60 seconds and retry. This only happens once per class per container lifetime. On Render free tier (15 min idle timeout), this will happen again after the container spins down.

### HTTP 500 / Internal Server Error

Check the response body for error details. Common causes:

- Missing `INDEX_URL` environment variable
- GitHub Releases URL changed or assets deleted
- Out of memory (try accessing fewer classes)

### Images return HTTP 404

The image filename might not exist in the archive. Possible reasons:

- The original PNG file was zero bytes (excluded during conversion)
- The image prefix maps to an unmapped class (rare edge case)
- The class archive hasn't finished downloading yet (retry after 30s)

### Server not responding

Render free tier spins down after 15 minutes of inactivity. The health endpoint may take 30–60s to respond on cold start. This is normal.

---

## Repository Contents

| File | Description |
|------|-------------|
| `server.py` | FastAPI server with all endpoints, image lazy loading |
| `Dockerfile` | Python 3.12-slim Docker image |
| `render.yaml` | Render Blueprint for deployment |
| `requirements.txt` | Python dependencies |
| `ncert_rag.ipynb` | Colab notebook for pipeline (download → extract → chunk → embed) |
| `chapter_index.json` | Full NCERT textbook index by class/subject/chapter |
| `merged.json` | All NCERT content merged into a single structured JSON reference (~1.1 MB) |
| `README.md` | This file |
