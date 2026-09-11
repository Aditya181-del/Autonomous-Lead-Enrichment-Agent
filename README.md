# Autonomous Lead Enrichment Agent

An autonomous, evidence-grounded Python intelligence agent that extracts verified company intelligence (company overview, ICP/target audience, leadership team, deterministic contact points, and verified sources) from target company domains.

---

## 1. Overview

B2B sales, recruiting, and market research teams often need fast, reliable, and evidence-grounded company intelligence directly from first-party company websites. Traditional web scrapers either fail on modern JavaScript-rendered single-page applications or pass noisy HTML directly into large language models, resulting in high latency, hallucinated contacts, and broken output schemas.

This agent implements a robust, dual-path architecture:
- **Headless Browser Acquisition (Playwright)**: Reliably renders dynamic JavaScript SPAs.
- **Internal Relevance Discovery**: Discovers and ranks high-value pages (`/about`, `/team`, `/contact`, `/pricing`) while filtering out irrelevant or malicious lookalikes.
- **Dual-Path Evidence Pipeline**:
  - *Deterministic Path*: Regex-based extraction of exposed emails (text + `mailto:`) and personal LinkedIn URLs (`/in/[^/?#]+`), preserving exact source attribution.
  - *Content Cleaner*: Strips scripts, styles, modals, banners, and boilerplate to reduce token overhead by up to 90%.
- **Evidence-Grounded Semantic Extraction (Ollama)**: Uses local Ollama models (e.g., `nemotron-3-super:cloud`, `llama3`) with strictly enforced Pydantic JSON schemas and `temperature=0`.
- **Application Validation Layer**: Verifies that emitted sources correspond to actual acquired pages and validates person names/roles against evidence.
- **Explainable Confidence Scoring**: A deterministic heuristic score (0.00 – 1.00) measuring crawl coverage, field presence, leadership grounding, and source attribution.
- **Resilience & Batch Concurrency**: Bounded concurrency with `asyncio.Semaphore`, retry mechanisms on transient errors, and domain-level failure isolation.

---

## 2. Architecture & Pipeline

```text
Target Domains (CLI / Input)
       ↓
Domain Normalization (e.g., https://www.postman.com/ -> postman.com)
       ↓
Headless Browser Acquisition (Playwright / Chromium)
       ↓
Homepage Fetch + Internal URL Discovery + Subdomain Filtering
       ↓
Semantic Relevance Scoring (Top-N candidate selection)
       ↓
Dual-Path Page Processing:
   ├── [Deterministic Path] Extract emails & personal LinkedIn profiles
   └── [Cleaner Path] Strip HTML noise, modals, scripts, styles, boilerplate
       ↓
Evidence Assembly (Cleaned text tagged with URLs and titles)
       ↓
Ollama Semantic Extraction (Dedicated LLMCompanyExtraction schema)
       ↓
Application-Level Validation (Source verification, person name validation)
       ↓
Explainable Confidence Scoring (Deterministic 0.0 - 1.0 breakdown)
       ↓
Final Assembly -> CompanyIntelligence Model
       ↓
Machine-Readable Output (output/output.json & output/output.csv)
```

---

## 3. Key Design Decisions

### A. Why Playwright?
Modern SaaS and AI company websites (e.g. `postman.com`, `supabase.com`, `vapi.ai`) are built using client-side JavaScript frameworks (React, Next.js, Vue). Standard HTTP clients (`requests`, `httpx`) only receive empty shell HTML or hydration scripts. Playwright spins up headless Chromium, executes JavaScript, waits for DOM settlement, and captures fully rendered content and dynamically injected links.

### B. Why Relevance Discovery & Bounded Crawling?
Crawling every link on a website is slow, wasteful, and noisy. The discovery engine filters out external links and lookalike domains (e.g. `evilpostman.com`), excludes non-informational pages (`/privacy`, `/terms`, `/login`), and ranks relevant semantic paths (`/team`, `/about`, `/contact`, `/pricing`) using a hierarchical keyword scoring algorithm bounded by `MAX_PAGES_PER_DOMAIN`.

### C. Why Content Cleaning?
Raw HTML pages are bloated with SVG icons, CSS styles, JavaScript bundles, tracking scripts, navigation headers, and cookie consent modals. Passing raw HTML directly to an LLM wastes context tokens and degrades model attention. The cleaner uses BeautifulSoup and `lxml` to strip non-content elements and boilerplate, achieving substantial token reduction while preserving semantic paragraph and heading structure.

### D. Why Dual-Path Deterministic Extraction?
Emails and LinkedIn URLs are frequently found in headers, footers, or `mailto:` links—areas that semantic content cleaning might strip as navigation boilerplate. Deterministic regex extraction operates directly on the raw `PageEvidence`, guaranteeing zero hallucinations (no fabricated emails or LinkedIn profiles) while retaining source attribution.

### E. Why LLM Contract vs. Application Model Separation?
The LLM semantic contract (`LLMCompanyExtraction`) must be strictly decoupled from the final application contract (`CompanyIntelligence`):
- **LLM-Owned**: `company_overview`, `target_audience`, `leadership`, `sources`.
- **Application-Owned**: `domain`, deterministic `contact_points`, `confidence_score`, `crawl_status`, `error_message`.
Forcing the LLM to generate application metadata causes schema validation failures when the model omits or alters these fields.

### F. Application Validation Layer
LLM output is treated as untrusted data:
1. **Source Grounding**: Every source URL emitted by the LLM is checked against the set of actually acquired URLs. Unsupported or hallucinated URLs are rejected.
2. **Leadership Validation**: Filters out generic roles used as names (e.g. "CEO", "Founder", "Support Team") and removes LinkedIn URLs that were not observed in the actual page evidence.

### G. Explainable Confidence Scoring
Confidence is calculated via an explainable, deterministic heuristic formula (0.0 to 1.0) rather than an arbitrary model output:
- **Crawl & Homepage Foundation (0.25)**: Homepage 200 status and ratio of successfully acquired pages.
- **Company Overview Grounding (0.20)**: Presence and substance of a 2-sentence description.
- **Target Audience / ICP Grounding (0.15)**: Presence of actionable ICP description.
- **Leadership Evidence (0.20)**: Validated leaders with verified roles and grounded links.
- **Contact Information Evidence (0.10)**: Authoritative public generic emails discovered.
- **Source Attribution (0.10)**: Verified first-party evidence sources.

> [!NOTE]
> This confidence score is an evidence-based heuristic metric and is NOT a statistically calibrated probability.

### H. Anti-Prompt-Injection Defense
Scraped website content is treated as untrusted data. The LLM prompt encapsulates evidence between explicit boundary delimiters (`<<<BEGIN UNTRUSTED FIRST-PARTY WEBSITE EVIDENCE>>>` ... `<<<END UNTRUSTED FIRST-PARTY WEBSITE EVIDENCE>>>`) and enforces strict system rules instructing the model to never follow instructions embedded in web content.

---

## 4. Configuration

All configuration is managed via environment variables and `.env` using Pydantic Settings:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `APP_ENV` | `development` | Application environment (`development` / `production`) |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `LLM_PROVIDER` | `ollama` | LLM inference provider (`ollama`) |
| `LLM_MODEL` | `nemotron-3-super:cloud` | Model name deployed in Ollama (e.g. `llama3`, `qwen2.5`) |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama API endpoint host |
| `REQUEST_TIMEOUT_SECONDS` | `30` | Browser navigation timeout in seconds |
| `MAX_RETRIES` | `2` | Bounded retries for transient navigation failures |
| `MAX_PAGES_PER_DOMAIN` | `6` | Maximum internal candidate pages to scrape per domain |
| `MAX_CONCURRENT_DOMAINS` | `3` | Maximum concurrent domain crawls in batch mode |

---

## 5. Installation & Setup

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com/) running locally or accessible remotely
- Playwright Chromium browser binaries

### Steps

1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd autonomous-lead-enrichment-agent
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Playwright Chromium**:
   ```bash
   playwright install chromium
   ```

5. **Configure environment**:
   ```bash
   cp .env.example .env
   ```

6. **Ensure Ollama is running**:
   ```bash
   ollama pull nemotron-3-super:cloud
   # or
   ollama pull llama3
   ```

---

## 6. Usage & CLI

### Run with Default Test Set (`postman.com`, `supabase.com`, `vapi.ai`)
```bash
python -m app.main
```

### Run with Custom Company Domains
```bash
python -m app.main postman.com supabase.com vapi.ai stripe.com github.com
```

### Custom Output Paths
```bash
python -m app.main postman.com --json-output output/results.json --csv-output output/results.csv
```

---

## 7. Output Formats

### JSON Output (`output/output.json`)
```json
[
  {
    "domain": "postman.com",
    "company_overview": "Postman is an API platform for building and using APIs. It simplifies each step of the API lifecycle and streamlines collaboration.",
    "target_audience": "Software developers, API engineers, and enterprise development teams.",
    "leadership": [
      {
        "name": "Abhinav Asthana",
        "role": "CEO & Founder",
        "linkedin_url": null,
        "source_url": "https://www.postman.com/company/about-postman"
      }
    ],
    "contact_points": [
      {
        "email": "help@postman.com",
        "source_url": "https://www.postman.com/company/contact-us"
      }
    ],
    "confidence_score": 0.95,
    "sources": [
      "https://www.postman.com/company/about-postman",
      "https://www.postman.com/company/contact-us"
    ],
    "crawl_status": "success",
    "error_message": null
  }
]
```

### CSV Output (`output/output.csv`)
Tabular summary including:
- `domain`
- `confidence_score`
- `crawl_status`
- `company_overview`
- `target_audience`
- `leadership_count`
- `leadership_names`
- `contact_emails`
- `sources_count`
- `input_tokens`
- `output_tokens`
- `error_message`

---

## 8. Testing

Run the full pytest unit and integration test suite:

```bash
pytest -v
```

### Test Coverage Overview:
- `tests/test_models.py`: Model defaults, schema separation, Pydantic validation.
- `tests/test_cleaner.py`: Noise/boilerplate removal, line deduplication, whitespace normalization.
- `tests/test_discovery.py`: URL normalization, lookalike domain rejection, relevance ranking.
- `tests/test_extractor.py`: Deterministic email (`text` + `mailto:`) and personal LinkedIn `/in/` detection.
- `tests/test_validators.py`: Source URL grounding and leadership name/role validation.
- `tests/test_scoring.py`: Heuristic confidence scoring calculations across empty, partial, and full evidence states.
- `tests/test_crawler.py`: Browser crawler snapshot conversions, error status handling, and domain normalization.
- `tests/test_orchestrator.py`: Multi-domain batch execution, failure isolation, and JSON/CSV export.
- `tests/test_config.py`: Settings loading and default verification.

---

## 9. Telemetry & Cost Tracking

The agent captures exact model-reported usage metrics from Ollama:
- **`input_tokens` (`prompt_eval_count`)**: Exact prompt token count processed by the model.
- **`output_tokens` (`eval_count`)**: Exact completion token count generated by the model.
- **`total_tokens`**: Combined token volume.
- **`latency_seconds`**: High-precision wall-clock time spent on inference.

*(Because local Ollama deployments do not incur external API costs, telemetry focuses on exact token usage and latency rather than estimated dollar costs).*

---

## 10. Limitations

1. **Complex Bot Mitigation / CAPTCHAs**: Headless Playwright navigates dynamic JavaScript SPAs, but websites utilizing advanced Cloudflare Turnstile or reCAPTCHA challenges may block automated navigation.
2. **Contact Email Exposure**: The agent strictly extracts publicly displayed email addresses. If a company only exposes a webform without displaying an email, `contact_points` will be empty (no email addresses are hallucinated or guessed).
3. **LinkedIn Profile Discovery**: Only direct personal LinkedIn profile URLs (`/in/...`) explicitly linked on first-party pages are captured.

---

## 11. Experimental Layer: LangGraph & Streamlit Dashboard

In addition to the production CLI baseline, the repository includes an experimental **LangGraph-based state machine orchestrator** and a **Streamlit operational dashboard**.

### A. Why LangGraph?
LangGraph introduces an explicit, typed state machine (`EnrichmentState`) with:
- **Observable execution stages**: `initialize` → `acquire_evidence` → `prepare_evidence` → `evaluate_evidence` → `extract_semantics` → `validate_result` → `score_result` → `finalize`.
- **Conditional Routing & Bounded Recovery**: Evaluates evidence completeness and routes to `recover_evidence` if needed (bounded by `max_recovery_attempts = 1`) or advances to semantic extraction.
- **Service Reuse**: All deterministic scraping, cleaning, validation, and scoring remain isolated in existing stateless services.

```text
Streamlit Dashboard (app/ui.py)
        ↓
LangGraph Orchestrator (app/graph.py)
        ↓
Existing Modular Services (crawler, discovery, cleaner, extractor, Ollama, validators, scoring)
        ↓
Output Artifacts (output/output.json & output/output.csv)
```

### B. Launching the Streamlit UI
```bash
streamlit run app/ui.py
```

### C. UI Features
- **Multi-Domain Input**: Text area supporting arbitrary batch company domains.
- **Engine Selection**: Toggle between `LangGraph Agentic Graph` and `Baseline Production Pipeline`.
- **Live Progress & Streaming Logs**: Stage-level status progression and real-time logs.
- **Interactive Result Cards**: Formatted summaries, leadership profiles, deterministic emails, source links, and raw JSON inspectors.
- **Export**: One-click download buttons for `output.json` and `output.csv`.

### D. Technical Demonstration Video
A full video recording demonstrating concurrent LangGraph execution on `postman.com`, `supabase.com`, and `vapi.ai` inside the Streamlit dashboard is provided in the repository at:
- [`demo/lead_enrichment_agent_demo.mp4`](demo/lead_enrichment_agent_demo.mp4)
- [`demo/README.md`](demo/README.md)