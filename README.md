# NCERT RAG API

Semantic search over all NCERT textbooks (Classes 1–12, all subjects). 19,678 indexed chunks across 1,089 chapters, with 82,124 embedded images. Deployed as a zero-dependency API — no API keys needed at runtime.

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────┐
│  AI Agent   │────▶│  FastAPI Server  │────▶│  FAISS Index │
│  (any tool) │     │  (Render Free)   │     │  (384-dim)   │
└─────────────┘     │  fastembed/      │     │  FlatIP      │
                    │  ONNX CPU        │     └──────────────┘
                    └──────────────────┘
                           │
                    ┌──────┴──────┐
                    │  GitHub     │
                    │  Releases   │
                    │  (v1: index │
                    │   + meta)   │
                    └─────────────┘
```

- **Embedding**: `sentence-transformers/all-MiniLM-L6-v2` (384-dim) via fastembed ONNX runtime.
- **Index**: FAISS `IndexFlatIP` (inner product = cosine similarity on normalized vectors).
- **Cold start**: Downloads `faiss.index` (28.8 MB) + `chunks_meta.pkl` (34.8 MB) + `images.tar.gz` (1018 MB) from GitHub Releases, caches in `/tmp/ncert_rag`.
- **Images**: 79,617 textbook diagrams/photos compressed as JPEG, served on demand from archive.
- **No API keys**: All embedding is local.

## API Endpoints

### `GET /health`

Server status and index stats.

```json
{"status":"ok","vectors":19678,"chunks":19678,"images_ready":true}
```

### `GET /v1/rag/chapters`

List every unique chapter across all textbooks.

```json
{
  "object": "rag.chapters",
  "total": 1089,
  "chapters": [
    {"std": "1", "subject": "Mathematics", "textbook": "Joyful Mathematics-1", "chapter": "Finding the Furry Cat! (Pre-number Concepts)", "chapter_number": "1"},
    ...
  ]
}
```

**Fields**
| Field | Type | Description |
|-------|------|-------------|
| `std` | string | Class (1–12) |
| `subject` | string | Subject name (e.g. Physics, Chemistry, History) |
| `textbook` | string | Textbook title |
| `chapter` | string | Chapter title |
| `chapter_number` | string | Chapter number |

### `POST /v1/rag/query`

Semantic search over all textbook chunks. Returns the most relevant passages for a free-text query.

**Request**
```json
{"query": "photosynthesis", "top_k": 5}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `query` | string | required | Natural language query |
| `top_k` | int | 5 | Number of results (max 19,678) |

**Response**
```json
{
  "object": "rag.query",
  "query": "photosynthesis",
  "total_results": 2,
  "results": [
    {
      "text": "PHOTOSYNTHESIS IN HIGHER PLANTS\n151\nSUMMARY\nGreen plants make their own food...",
      "source": {
        "std": "11",
        "subject": "Biology",
        "textbook": "Biology",
        "chapter": "Photosynthesis in Higher Plants",
        "chapter_number": "11",
        "pages": [21]
      },
      "relevance_score": 0.7623,
      "image_count": 3,
      "image_urls": [
        "/v1/rag/image/gegp108_p20_0.jpg",
        "/v1/rag/image/gegp108_p20_1.jpg"
      ]
    }
  ]
}
```

**Fields**
| Field | Type | Description |
|-------|------|-------------|
| `text` | string | Retrieved passage text |
| `source.std` | string | Class level |
| `source.subject` | string | Subject name |
| `source.textbook` | string | Textbook title |
| `source.chapter` | string | Chapter title |
| `source.chapter_number` | string | Chapter number |
| `source.pages` | list[int] | Page numbers in source PDF |
| `relevance_score` | float | Cosine similarity (0–1) |
| `image_count` | int | Images embedded in this chunk |
| `image_urls` | list[str] | Relative URLs to serve images for this chunk |

### `POST /v1/rag/chapter`

Retrieve all chunk content for a chapter by name. Supports optional filters by `std` and `subject`.

**Request**
```json
{"chapter": "matter", "std": "11", "subject": "Physics"}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `chapter` | string | required | Chapter name (case-insensitive, partial match) |
| `std` | string | null | Filter by class (e.g. "11", "9") |
| `subject` | string | null | Filter by subject (exact, case-insensitive) |

**Response**
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

**Fields**
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
| `matches[].chunks[].image_count` | int | Images in this chunk |
| `matches[].chunks[].image_urls` | list[str] | Relative URLs to serve images for this chunk |

**Filtering examples**

| Request | Effect |
|---------|--------|
| `{"chapter": "matter"}` | All chapters matching "matter" across all classes |
| `{"chapter": "matter", "std": "11"}` | Only class 11 chapters matching "matter" |
| `{"chapter": "matter", "subject": "Physics"}` | Only Physics chapters matching "matter" |
| `{"chapter": "matter", "std": "11", "subject": "Physics"}` | Class 11 Physics chapters matching "matter" |

### `GET /v1/rag/image/{filename}`

Serve a textbook image by filename. Images are extracted from PDFs, converted to JPEG, and served from a compressed archive.

**Example**

```bash
# Fetch a specific image from a query result
curl https://ncert-rag-api.onrender.com/v1/rag/image/gegp108_p20_0.jpg -o diagram.jpg
```

| Field | Description |
|-------|-------------|
| `filename` | Image filename from `image_urls` list in query/chapter responses |

Returns `image/jpeg` binary content, or `404` if the image is not in the archive.

> **Note**: Images are served lazily — first request for a class triggers a background download + extract (~30-60s). Subsequent requests serve instantly from `/tmp` cache. Each class archive is 35-133 MB compressed, extracted to disk on Render's ephemeral storage.

### Error responses

All endpoints return structured errors:

```json
{"detail": "No chapters matching 'xyznonexistent'"}
```

```json
{"detail": [{"type": "missing", "loc": ["body", "query"], "msg": "Field required"}]}
```

## Usage examples

### cURL

```bash
# Health check
curl https://ncert-rag-api.onrender.com/health

# List chapters
curl https://ncert-rag-api.onrender.com/v1/rag/chapters | jq '.total'

# Semantic query
curl -X POST https://ncert-rag-api.onrender.com/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query": "newton laws of motion", "top_k": 3}'

# Chapter lookup with filters
curl -X POST https://ncert-rag-api.onrender.com/v1/rag/chapter \
  -H "Content-Type: application/json" \
  -d '{"chapter": "force", "std": "9", "subject": "Science"}'
```

### Python

```python
import requests

BASE = "https://ncert-rag-api.onrender.com"

# Query
resp = requests.post(f"{BASE}/v1/rag/query", json={
    "query": "photosynthesis",
    "top_k": 3
})
data = resp.json()
for r in data["results"]:
    s = r["source"]
    print(f"[{s['std']} {s['subject']} ch{s['chapter_number']}] "
          f"{s['chapter']} | score: {r['relevance_score']}")
    # Fetch images
    for img_url in r["image_urls"]:
        img_data = requests.get(f"{BASE}{img_url}").content
        # save or process img_data

# Chapter lookup
resp = requests.post(f"{BASE}/v1/rag/chapter", json={
    "chapter": "matter",
    "std": "11",
    "subject": "Physics"
})
data = resp.json()
for m in data["matches"]:
    print(f"{m['chapter']}: {m['total_chunks']} chunks")
    # First chunk's images
    for chunk in m["chunks"][:1]:
        for img_url in chunk["image_urls"]:
            print(f"  Image: {img_url}")
```

## Deployment

The API runs on **Render Free** (Docker, Python 3.12-slim). Cold start takes ~30s (model + index download).

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

### Manual deploy

```bash
curl -X POST https://api.render.com/v1/services/<SERVICE_ID>/deploys \
  -H "Authorization: Bearer $RENDER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"clearCache": "clear"}'
```

### Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `INDEX_URL` | Yes | Base URL for `faiss.index`, `chunks_meta.pkl`, `images.tar.gz` |
| `IMAGES_URL` | No | Base URL for `images.tar.gz` (falls back to `INDEX_URL`) |
| `PORT` | No | Server port (default 8000) |

## Dataset

| Stat | Value |
|------|-------|
| PDFs processed | 1,091 |
| Total chunks | 19,678 |
| Total chapters | 1,089 |
| Total images extracted | 82,124 |
| Embedding dimension | 384 |
| Classes covered | 1–12 |
| Subjects | 27 (Math, Physics, Chemistry, Biology, History, Geography, Economics, etc.) |

## Pipeline

The index was built in 4 phases on Google Colab:

1. **Download**: Fetch 1,091 PDF NCERT textbooks
2. **Extract**: PyMuPDF text + image extraction per page
3. **Chunk**: Section-based chunking (~500 tokens each)
4. **Embed**: `all-MiniLM-L6-v2` via sentence-transformers → FAISS index
