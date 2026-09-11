# Technical Demonstration — Autonomous Lead Enrichment Agent

This directory contains the technical demonstration walkthrough and video recording of the **Autonomous Lead Enrichment Agent** running the experimental **LangGraph Agentic Graph** inside the **Streamlit Operational Dashboard**.

---

## 1. Demo Video Recording

- **File**: [`lead_enrichment_agent_demo.mp4`](./lead_enrichment_agent_demo.mp4) / [`lead_enrichment_agent_demo.webp`](./lead_enrichment_agent_demo.webp)
- **Target Domains Tested**:
  - `postman.com`
  - `supabase.com`
  - `vapi.ai`
- **Configuration**: Concurrency = 3, Max Pages Per Domain = 6, Timeout = 30s
- **Model**: `nemotron-3-super:cloud` (via local Ollama at `http://localhost:11434`)

---

## 2. Walkthrough Stages

### Stage 1: Setup & Input Configuration
- Launching `python -m streamlit run app/ui.py`.
- Configured with bounded concurrency (`asyncio.Semaphore`), relevance ranking, and structured schema extraction.

### Stage 2: Concurrent LangGraph Execution
- **State Machine**: `initialize` ➔ `acquire_evidence` ➔ `prepare_evidence` ➔ `route_evidence_check` ➔ `extract_semantics` ➔ `validate_result` ➔ `score_result` ➔ `finalize`.
- Real-time stage logging and progress tracking across all 3 domains concurrently.

### Stage 3: Results & Evidence Grounding
- **Metrics**: 3 / 3 Domains Enriched | Avg Confidence: **0.90** | Total Tokens: **29,227** | Total Runtime: **151.85s**.
- **Extracted Intelligence**:
  - **`postman.com`** (Score: 0.90): Overview, Developer/Enterprise ICP, Leadership team, and 3 deterministic contact emails (`help@postman.com`, `info-jp@postman.com`, `info@postman.com`).
  - **`supabase.com`** (Score: 0.90): Overview, Developer ICP, Leadership team, and 5 deterministic contact emails (`abuse@supabase.com`, `help-events@supabase.com`, `legal@supabase.com`, `privacy@supabase.com`, `security@supabase.com`).
  - **`vapi.ai`** (Score: 0.90): Voice AI overview, Developer/Enterprise ICP, Leadership team (`Jason Mitura`, `Jayson Noland`), and deterministic contact email (`talent@vapi.ai`).

### Stage 4: Artifact Downloads & Inspection
- Download buttons for `output/output.json` and `output/output.csv`.
- Interactive raw JSON and explainable heuristic confidence scoring breakdown inspectors.
