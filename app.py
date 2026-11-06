import os, io, uuid, requests
from fastapi import FastAPI, UploadFile
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from pypdf import PdfReader

app = FastAPI()
emb = TextEmbedding("BAAI/bge-small-en-v1.5")
qd = QdrantClient(url="http://qdrant:6333")
OLLAMA = "http://ollama:11434"
COL = "docs"
if not qd.collection_exists(COL):
    qd.create_collection(COL, vectors_config=VectorParams(size=384, distance=Distance.COSINE))

def chunk(t, size=800, overlap=100):
    return [t[i:i+size] for i in range(0, len(t), size-overlap)]

@app.post("/ingest")
async def ingest(file: UploadFile):
    data = await file.read()
    text = "".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(data)).pages)
    chunks = chunk(text)
    vecs = list(emb.embed(chunks))
    qd.upsert(COL, [PointStruct(id=str(uuid.uuid4()), vector=v.tolist(),
              payload={"text": c}) for c, v in zip(chunks, vecs)])
    return {"chunks": len(chunks)}

@app.post("/ask")
def ask(q: str, k: int = 4):
    v = list(emb.embed([q]))[0].tolist()
    hits = qd.query_points(COL, query=v, limit=k).points
    ctx = "\n---\n".join(h.payload["text"] for h in hits)
    prompt = f"Answer only from this context. If unknown, say so.\n\n{ctx}\n\nQuestion: {q}"
    r = requests.post(f"{OLLAMA}/api/generate", timeout=300,
        json={"model": "llama3.2:1b", "prompt": prompt, "stream": False})
    return {"answer": r.json()["response"],
            "sources": [h.payload["text"][:150] for h in hits]}
