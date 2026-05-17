# Use Python 3.12 slim
FROM python:3.12-slim

WORKDIR /app

# Install system deps for faiss
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends gcc g++ && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .

# Volume for cached index files
VOLUME /data

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
