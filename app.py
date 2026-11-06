import io, os, time, uuid, requests
from fastapi import FastAPI, UploadFile
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from pypdf import PdfReader

app = FastAPI()
emb = TextEmbedding("BAAI/bge-small-en-v1.5")
qd = QdrantClient(url="http://qdrant:6333")
COL = "docs"
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)

if not qd.collection_exists(COL):
    qd.create_collection(
        COL, vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )


def chunk(t, size=800, overlap=100):
    return [t[i:i + size] for i in range(0, len(t), size - overlap)]


@app.post("/ingest")
async def ingest(file: UploadFile):
    data = await file.read()
    text = "".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(data)).pages)
    chunks = chunk(text)
    vecs = list(emb.embed(chunks))
    qd.upsert(
        COL,
        [
            PointStruct(id=str(uuid.uuid4()), vector=v.tolist(), payload={"text": c})
            for c, v in zip(chunks, vecs)
        ],
    )
    return {"chunks": len(chunks)}


@app.post("/ask")
def ask(q: str, k: int = 4):
    v = list(emb.embed([q]))[0].tolist()
    hits = qd.query_points(COL, query=v, limit=k).points
    sources = [h.payload["text"] for h in hits]
    ctx = "\n---\n".join(sources)
    prompt = (
        "You are a strict assistant. Use ONLY the context below. "
        "If the context does not contain the exact answer, reply exactly: I don't know.\n"
        "Keep the answer to one short sentence and include exact terms from the context.\n\n"
        f"Context:\n{ctx}\n\nQuestion: {q}\nAnswer:"
    )
    answer = "LLM error: no response"
    for attempt in range(3):
        data = None
        try:
            r = requests.post(
                GEMINI_URL,
                headers={"x-goog-api-key": GEMINI_KEY, "Content-Type": "application/json"},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0},
                },
                timeout=90,
            )
            data = r.json()
            answer = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            break
        except requests.exceptions.RequestException as e:
            answer = f"LLM error: {type(e).__name__}"
            time.sleep(3)
        except (KeyError, IndexError, ValueError):
            answer = f"LLM error: {data}"
            time.sleep(5)
    return {"answer": answer, "sources": sources}
