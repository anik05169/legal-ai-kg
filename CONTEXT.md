# 🏛️ Legal AI GraphRAG - Current Project Context & State

> [!IMPORTANT]
> **INSTRUCTIONS FOR AI ASSISTANTS (LLMs):**
> 1. **Read this file FIRST** before writing code, modifying configurations, or formulating implementation plans.
> 2. **Update this file LAST** before completing your turn whenever you make changes to the repository. Add a log entry in the **Recent Implementation Log**, update the **Current Project State** or **Directory Architecture** if files were moved/added, and adjust the **Backlog & Next Steps** accordingly.

---

## 🏁 Current Project State

* **Active Phase**: `Phase 2: Modular Cleanup & Organization`
* **Current Stage**: The codebase has been completely refactored from a monolithic flat structure into a clean, modular architecture. Core logical modules reside in `core/`, experimental scripts and test runners reside in `tests/`, and web dashboard static assets reside in `static/` and `lib/`. The codebase is linked and fully synced to the remote repository **`https://github.com/anik05169/legal-ai-kg`**.
* **Last Reorganization Date**: 2026-05-20
* **API Credential Policy**: Fully secured. Hardcoded OpenAI API keys have been removed from `config.py` and replaced with dynamic environment lookups (`os.getenv("OPENAI_API_KEY")`). Storing credential variables in `.env` is supported and configured under git ignore policies.

---

## 📂 Directory Architecture

Below is the verified, organized file structure of the **`legal-ai-kg`** repository:

```text
legal-ai-kg/
├── 📁 core/                           # <-- Core pipeline & logic engine
│   ├── config.py                      # Models, API keys, thresholds, and taxonomy constraints
│   ├── data_loader.py                 # CUAD dataset ingestion layer
│   ├── kg_indexer.py                  # Text chunker, GLiNER NER, GLiREL relation extractor, DB builder
│   ├── rag_pipeline.py                # Semantic VDB retrieval + 1-hop graph expansion & LLM synthesis
│   └── visualize_kg.py                # PyVis network rendering engine (Vis.js visualization)
├── 📁 tests/                          # <-- Testing utilities and experimental code
│   ├── ner_and_relation.py            # GLiNER + GLiREL token extraction sandbox
│   ├── random_contract.py             # Random CUAD contract & QA previewer
│   ├── test_edges.py                  # Graph JSON edge-relation reader
│   ├── test_load.py                   # PyTorch hardware accelerator & core model loader
│   ├── test_rag.py                    # Vector DB & Graph metadata integration tester
│   ├── test_rag_debug.py              # Semantic matching & edge intersection analyzer
│   └── test_rag_query.py              # CLI context & triplet query runner
├── 📁 static/                         # Web dashboard assets (HTML, CSS, custom JS)
├── 📁 lib/                            # Frontend visualizer library files (vis-network, tom-select)
├── app.py                             # Root FastAPI server entrypoint (serves static UI & streams SSE logs)
├── main.py                            # Root CLI orchestrator loop (ingests, visualizes, and queries via terminal)
├── README.md                          # Detailed developer README with system architecture & setup instructions
├── requirements.txt                   # Project package dependencies
├── .gitignore                         # Version control exclusions (ignores .env, chroma_db, html output, json graph)
└── CONTEXT.md                         # <-- THIS FILE (Living context document for AI assistants)
```

---

## 🛠️ Technical Specifications & Setup

1. **Extraction Engine**: 
   * Named Entity Recognition: **GLiNER** (`urchade/gliner_medium-v2.1`)
   * Relationship Extraction: **GLiREL** (`jackboyla/glirel-large-v0`)
   * Validation System: Strict relation mapping in `config.py` (`VALID_RELATIONS`) filters out spurious or illogical triplets (e.g. `Location` to `Date` connections).
2. **Vector Space Database**:
   * Storage: **ChromaDB** persistent engine residing locally in `./chroma_db`.
   * Embedder: **SentenceTransformers** (`BAAI/bge-small-en-v1.5`).
3. **Knowledge Graph System**:
   * Logical Layer: **NetworkX** directed graph (`DiGraph`) saved and loaded locally as `./legal_kg.json`.
   * Visual Layer: **PyVis / Vis.js** compiling nodes (sized by degree, colored by entity labels) into `./interactive_graph.html`.
4. **LLM Engine**:
   * OpenAI **GPT-4o-Mini** synthesizing semantic matching context and structural graph facts.

---

## 📝 Recent Implementation Log

### 📅 2026-05-20
* **Organized Repository Structure**: Cleaned up the root directory by creating `core/` and `tests/` subdirectories and moving corresponding files.
* **Import Refactoring**: 
  * Updated `app.py` and `main.py` root entrypoints to import core modules via `core.kg_indexer`, `core.rag_pipeline`, etc.
  * Converted all internal core file imports to package-relative imports (`from . import config`).
  * Modified test files to dynamically add parent directory to `sys.path` for robust, path-independent imports.
* **Living Document Creation**: Added `CONTEXT.md` to serve as a persistent system context file for downstream AI workflows.
* **GitHub Sync**: Pushed all organized commits to `origin main` successfully.

### 📅 2026-05-17
* **Standalone Repo Setup**: Cloned files to new clean repository `legal-ai-kg`.
* **API Credentials Security**: Removed raw hardcoded OpenAI API keys from `config.py` to prevent security scanning violations. Configured dynamic lookup via `os.getenv` and `.env` files.
* **GitHub Setup & Initial Push**: Initialized git repo, added customized `.gitignore` excluding databases and visual HTML outputs, and pushed clean branch to remote `https://github.com/anik05169/legal-ai-kg.git`.
* **Documentation Build**: Generated a detailed, comprehensive system architecture README with step-by-step guides.

---

## 🎯 Backlog & Next Steps

- [ ] **Local Filesystem Ingestion**: Swap out the Hugging Face CUAD downloader with a local directory loader (e.g., parsing a `contracts_vault/` folder).
- [ ] **Delta Indexing**: Implement delta-indexing checks (only parsing new or modified files rather than wiping and rebuilding the entire graph).
- [ ] **Neo4j DB Integration**: Switch from in-memory NetworkX JSON serialization to a live Neo4j database once the document index grows beyond ~500 contracts.
- [ ] **Advanced Entity Resolution**: Implement link resolution and node deduplication algorithms to prevent node naming collision across multiple corporate contracts.
- [ ] **Multi-hop Cypher Queries**: Build Cypher templates for advanced, deep structural relationship searches.
