# Smart Legal Contracts

Upload a contract. Get a classified risk breakdown. Ask GPT-4 to rewrite the hostile clauses without losing legal intent. Built around Legal-BERT embeddings and a curated clause library.

## What it does

- Parse a contract (PDF or DOCX) into clauses
- Classify each clause: payment terms, indemnification, confidentiality, termination, etc.
- Score each clause against a library of known-friendly and known-hostile variants using cosine similarity on Legal-BERT embeddings
- Surface the top-N riskiest clauses with plain-English explanations
- Offer GPT-4 rewrites that preserve legal intent and call out the tradeoff

## Tech

- Python 3.11, FastAPI, Pydantic v2
- Legal-BERT (`nlpaueb/legal-bert-base-uncased`) for domain-specific embeddings
- sentence-transformers for the classification head
- PostgreSQL + pgvector for the clause library (ivfflat cosine)
- Streamlit for the demo UI; separate production frontend in the `frontend/` integration guide
- Docker, deployed on Cloud Run
- Full test coverage reported in coverage/

## Architecture

See [ARCHITECTURE_REVIEW.md](ARCHITECTURE_REVIEW.md) and [`docs/RAG Architecture for Legal Analysis/`](docs/) for the full rationale: why Legal-BERT over general embeddings, why a curated clause library instead of zero-shot, where LLMs help and where they don't.

## Run locally

```bash
git clone https://github.com/rahulmehta25/Smart-Legal-Contracts
cd Smart-Legal-Contracts
cp .env.example .env
# set OPENAI_API_KEY and POSTGRES_* values
docker compose up -d postgres
make install
make run-backend  # FastAPI on http://localhost:8000
```

`make help` lists all available targets.

## Status

Prototype quality. Clause library seeded from public templates and curated adversarial examples. Not a substitute for a real lawyer, but the classifier agrees with human review on the samples we tested.

MIT licensed.
