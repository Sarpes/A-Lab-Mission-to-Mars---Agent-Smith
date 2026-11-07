# a-lab-chatbot-casting

Agent Smith: a Turkish customer chatbot for the fictional company "Mars Turizm" (A-Lab chatbot casting assignment). It answers company questions from a document with RAG (retrieval-augmented generation: it looks up matching lines first) and never invents facts. Local LLM through Ollama, Streamlit UI.

## Layout
- `app.py` Streamlit UI; `bot.py` chat logic; `rag.py` retrieval; `ayarlar.py` settings from `.env`; `kilit.py` lock; `kuyruk.py` queue.
- `persona.txt` the live persona; `data/` the source document and search terms; `eval/` retrieval questions; `tests/` unit tests; `tools/` WebSocket checks.
- `docs/` is git-ignored on purpose (working notes, removed from the repo until after the demo, 10.11.2026). Its workspace subfolders (handovers, architecture, guides, artifacts, archive) exist locally only.

## Commands
- Run: `python -m streamlit run app.py` (needs Ollama 0.40+ with `gemma4:latest` and `bge-m3` pulled).
- Tests: `python -m pytest -q` (66 unit tests, no Ollama needed).
- Retrieval measurement: `python eval_rag.py` (needs Ollama + bge-m3).

## Remotes
- `origin` = Sarpes/A-Lab-Mission-to-Mars---Agent-Smith: push here.
- `alab` = the team repo: never push without Sarp's explicit OK.

## Never
- Never commit `.env` (it holds the access password); only `.env.example`.
- Never commit straight to `main`: one branch per task.
