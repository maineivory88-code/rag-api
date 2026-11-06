
cat > README.md << 'EOF'
# RAG API with Docker, Qdrant and Gemini

A containerized Retrieval-Augmented Generation API. Upload a PDF, ask questions, get answers grounded in the document, with an eval script to measure quality.

## Architecture

PDF -> chunking -> fastembed embeddings -> Qdrant vector DB -> top-k retrieval -> Gemini -> answer + sources

- FastAPI service (app.py)
- Qdrant vector database
- Docker Compose to run both
- eval.py to score retrieval and answers

## Run it

1. Get a free API key at aistudio.google.com
2. Clone the repo and create your env file:

        git clone https://github.com/YOUR_USERNAME/rag-api.git
        cd rag-api
        cp .env.example .env
        (edit .env and paste your key)

3. Start the stack:

        docker compose up --build -d

4. Open http://localhost:8000/docs
5. POST /ingest with a PDF (once), then POST /ask with a question.

On GitHub Codespaces, open the Ports tab, tap the globe next to port 8000 and add /docs to the URL.

## Evaluate

Edit questions.json with your own questions and expected keywords, then run:

    python3 eval.py

It reports retrieval hit rate, answer accuracy and correct refusals.

## Results

| Run | Chunk size | k | Retrieval | Answer | Refusal |
|---|---|---|---|---|---|
| 1 | 800 | 4 | _/9 | _/9 | _/0 |

## Notes

- The API retries and falls back across Gemini models when one is overloaded.
- Never commit your .env file.
- Free-tier prompts may be used by Google to improve its products, so don't upload private documents.
EOF
