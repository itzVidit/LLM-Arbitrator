# ⚖️ LLM Output Arbitration System

> A multi-agent AI evaluation pipeline that routes any LLM response through **five independent specialist critics** running on different models and providers, then synthesises their findings into a single confidence-scored verdict with inline annotations and a final APPROVE / REVISE / REJECT recommendation.

<div align="center">

### Why trust one AI to judge another AI's work?

The LLM Arbitration System uses an **ensemble of five diverse models**, each specialised for a different failure mode, to evaluate responses the way a panel of expert reviewers would — then arbitrates their disagreements to reach a fair, explainable verdict.

```
✨ Five critics. One verdict. Zero bias. ✨
```

</div>

---

## 📋 Table of Contents

- [What Problem Does This Solve?](#-what-problem-does-this-solve)
- [Core Concept: What Is LLM Arbitration?](#-core-concept-what-is-llm-arbitration)
- [The Five Critics](#-the-five-critics)
- [System Architecture](#-system-architecture)
- [How Each Component Works](#-how-each-component-works)
- [Data Flow: From Request to Verdict](#-data-flow-from-request-to-verdict)
- [Project Structure](#-project-structure)
- [Setup and Running](#-setup-and-running)
- [Using the API](#-using-the-api)
- [Streamlit Dashboard](#-streamlit-dashboard)
- [Configuration Reference](#-configuration-reference)
- [Cost](#-cost)
- [Tech Stack](#-tech-stack)
- [Future Work](#-future-work)

---

## 📌 What Problem Does This Solve?

Large Language Models hallucinate. They give confident wrong answers, miss parts of the question, introduce unsafe content, and write in inconsistent styles — often all in the same response.

The standard fix is either:
- **Self-evaluation** — ask the same model to check its own work (it shares its own blind spots)
- **Human review** — accurate but slow and expensive at scale

This project takes a third approach: **an ensemble of five diverse AI models, each specialised for a different type of error, evaluating the same response independently — and then having their disagreements arbitrated to reach a final verdict.**

The result is a structured, explainable output with:
- A **0–100 score** and **APPROVE / REVISE / REJECT** recommendation
- **Colour-coded inline annotations** on every flagged phrase
- **Per-dimension breakdowns** with evidence and fix recommendations
- An **adjudicator** that resolves conflicts between critics when they disagree

---

## 🧠 Core Concept: What Is LLM Arbitration?

**Arbitration** in this context means running a response through multiple independent evaluators (critics) and then adjudicating their disagreements to reach a final verdict — the same way a panel of judges works.

Each critic runs on a **different model from a different provider**, evaluating a different quality dimension. This is intentional: a model that excels at fact-checking may miss logical fallacies; a safety-specialised model will catch policy violations that a general-purpose model overlooks.

> **Diversity in evaluation = more reliable verdicts.**

The pipeline has two routing paths depending on how much the critics agree:

- **Fast path** — all critics agree and all scores are high → skip the adjudicator, synthesise immediately
- **Disagreement path** — score variance exceeds the threshold → route to the adjudicator LLM, which resolves each conflict with dimension-specific evidence-based reasoning, then synthesises



---

## 🔬 The Five Critics

| # | Critic | Model | Provider | Specialisation |
|---|--------|-------|----------|----------------|
| 1 | **Accuracy** | Gemini 2.5 Flash | Google AI Studio | Fact-checking, hallucination detection, contradictions |
| 2 | **Logic** | LLaMA 3.3 70B Versatile | Groq | Reasoning chains, logical fallacies, non-sequiturs |
| 3 | **Completeness** | Gemma 3 | Ollama (local) | Whether the prompt's requirements were fully addressed |
| 4 | **Safety** | Qwen3 235B | OpenRouter (free) | Harmful content, bias, prompt injection, policy violations |
| 5 | **Style** | Gemma 3 | Ollama (local) | Grammar, clarity, tone, formatting consistency |

Each critic returns a structured `Critique` object containing:
- A **0–100 score** for its dimension
- A **confidence float** (0.0 – 1.0) in its own assessment
- A list of **`Issue` objects**, each with: an exact quote, explanation, severity (CRITICAL / MAJOR / MINOR / INFO), evidence, and a fix recommendation
- A **pass / fail** flag (fails if any CRITICAL or MAJOR issue was found)
- The **model name** used and **latency in ms**

Every critic runs **in parallel** using LangGraph's `Send` API — the total pipeline time is approximately the slowest single critic, not the sum of all five.

---

## 🏗️ System Architecture

The pipeline is a **directed acyclic graph (DAG)** compiled with LangGraph. Here is the full topology:

```
                    ┌─────────────────────────────┐
                    │       User  /  REST API      │
                    │  { original_prompt,          │
                    │    llm_response }            │
                    └──────────────┬──────────────┘
                                   │
                          ┌────────▼────────┐
                          │   parse_input   │  Validates ArbitrationRequest
                          └────────┬────────┘
                                   │
               ╔═══════════════════╧═══════════════════╗
               ║   Fan-out via LangGraph Send API       ║
               ╠════════════════════════════════════════╣
       ┌───────▼──┐  ┌───────▼──┐  ┌───────▼──┐  ┌───────▼──┐  ┌───────▼──┐
       │ Accuracy │  │  Logic   │  │Complete. │  │  Safety  │  │  Style   │
       │ (Gemini) │  │  (Groq)  │  │ (Ollama) │  │(OpenRtr) │  │ (Ollama) │
       └───────┬──┘  └───────┬──┘  └───────┬──┘  └───────┬──┘  └───────┬──┘
               └─────────────┴─────────────┴─────────────┴─────────────┘
                                   │  Fan-in — all 5 Critique objects collected
                          ┌────────▼────────┐
                          │detect_disagreem.│  Computes score std-dev,
                          └────────┬────────┘  pass/fail conflicts,
                                   │           confidence outliers
                     ┌─────────────┴─────────────┐
                     │                           │
              [variance > 15.0             [all pass + scores
               OR pass/fail conflict]       above threshold]
                     │                           │
            ┌────────▼────────┐                  │
            │   adjudicate    │  Gemini resolves  │  ← FAST PATH
            │  (LLM node)     │  each conflict    │    (no adjudicator)
            └────────┬────────┘                  │
                     └─────────────┬─────────────┘
                          ┌────────▼────────┐
                          │synthesize_verdi.│  Builds FinalVerdict,
                          └────────┬────────┘  saves to SQLite
                                   │
                    ┌──────────────┴──────────────┐
                    │     FastAPI  │  Streamlit    │
                    └─────────────────────────────┘
```

### Routing Decision Logic

`detect_disagreement` measures:
- **Score variance** — standard deviation of the five critic scores
- **Pass/fail conflicts** — did critics disagree on whether the response passes?
- **Confidence outliers** — is one critic's confidence a statistical outlier?

If `score_variance > DISAGREEMENT_VARIANCE_THRESHOLD` (default 15.0) **or** there are pass/fail conflicts, the adjudicator is invoked. Otherwise the pipeline takes the fast path directly to synthesis.



---

## 🔧 How Each Component Works

### Critics (`critics/`)

Every critic inherits from `BaseCritic`, which provides:

- **Retry logic** — Tenacity retries 3× with exponential backoff on any exception
- **Timeout enforcement** — 30 s for remote APIs (Gemini, Groq, OpenRouter), 120 s for local Ollama cold-starts
- **Automatic fallback chain** — if the primary model fails, the critic transparently switches to a backup model/provider and tries once more before returning an error `Critique` so the rest of the pipeline can continue

```
Accuracy critic primary call fails (404 / timeout)
    ↓
Switch to fallback: LLaMA 3.3 70B on Groq
    ↓ (if that also fails)
Return Critique(score=0, passed=False, summary="critic unavailable: …")
Pipeline continues with the four remaining critiques
```

Each critic only needs to implement `_build_prompt(request)`. Everything else — client setup, retry, timeout, fallback, Pydantic coercion — is handled by the base class.

**Ollama special case:** local models (Gemma 3) don't reliably follow Instructor's JSON schema injection. `BaseCritic._ollama_call()` injects a flat, human-readable JSON template directly into the system prompt and parses the response manually before passing it through `Critique.model_validate()`.

---

### Orchestration (`orchestration/`)

**`GraphState`** is a `TypedDict` carrying all data through the pipeline. Fields annotated with `Annotated[list, operator.add]` are merge-safe: every parallel critic node returns `{"critiques": [its_result]}` and LangGraph concatenates them — no race conditions, no overwrites.

**`graph.py`** builds the DAG:
```python
builder.add_conditional_edges("parse_input", dispatch_critics, critic_node_names)
# dispatch_critics returns [Send("run_accuracy_critic", state), Send("run_logic_critic", state), ...]
# LangGraph runs all five Sends concurrently in separate threads
```

**`adjudicator.py`** builds a detailed prompt that gives Gemini all five critiques, the `DisagreementReport`, and dimension-specific resolution instructions:
- **Accuracy** → fact-check every disputed claim with a source or known standard
- **Logic** → trace each reasoning step and name any fallacy explicitly
- **Completeness** → list every prompt requirement and whether it was addressed
- **Safety** → determine if the flagged content has a realistic harm vector or is a false positive
- **Style** → distinguish genuine quality problems from personal preference

The adjudicator returns an `AdjudicationResult` with confirmed issues, dismissed issues (with evidence-based reasons), per-dimension resolved scores, and an executive summary paragraph.

---

### Data Models (`models/critique.py`)

All LLM output is coerced into typed Pydantic v2 schemas via Instructor. The model hierarchy is:

```
ArbitrationRequest          ← what the caller sends in
       │
       ▼ (five parallel critics)
Critique × 5                ← one per critic; each contains list[Issue]
       │
       ▼
DisagreementReport          ← computed from the five Critiques
       │ (if disagreement)
       ▼
AdjudicationResult          ← adjudicator's structured resolution
       │
       ▼
FinalVerdict                ← the complete response returned to the caller
```

`Severity` and `CriticDimension` are `str` enums with `_missing_` overrides that normalise any casing from the LLM — `"CRITICAL"`, `"critical"`, `"Critical"` all map to the same value.

---

## 📊 Data Flow: From Request to Verdict

```
1. POST /v1/arbitrate  →  ArbitrateRequest (FastAPI)
2. routes.py           →  ArbitrationRequest (internal)
3. graph.invoke()      →  GraphState flows through DAG:
      parse_input
        → [5× critic nodes in parallel]
        → detect_disagreement
        → adjudicate (if needed)
        → synthesize_verdict
4. FinalVerdict        →  stored in SQLite (arbitration_id = UUID)
5. ArbitrateResponse   →  returned to caller (wire format)
```

Every stored verdict is queryable via `GET /v1/arbitrations/{id}` or browsable in the Streamlit dashboard.



---

## 📁 Project Structure

```
LLM Output Arbitration System/
│
├── .env                        ← API keys and model overrides (never commit)
├── config.py                   ← Typed Settings object via pydantic-settings
├── requirements.txt
├── test_critics.py             ← Smoke test runner
│
├── models/
│   └── critique.py             ← All Pydantic schemas: Issue, Critique,
│                                  ArbitrationRequest, DisagreementReport,
│                                  AdjudicationResult, FinalVerdict
│
├── critics/
│   ├── base.py                 ← BaseCritic: retry, timeout, fallback, Ollama handling
│   ├── accuracy.py             ← Gemini 2.5 Flash   (Google AI Studio)
│   ├── logic.py                ← LLaMA 3.3 70B      (Groq)
│   ├── completeness.py         ← Gemma 3            (Ollama — local)
│   ├── safety.py               ← Qwen3 235B         (OpenRouter free tier)
│   └── style.py                ← Gemma 3            (Ollama — local)
│
├── orchestration/
│   ├── state.py                ← GraphState TypedDict (LangGraph state)
│   ├── graph.py                ← DAG: nodes, edges, fan-out via Send API
│   ├── nodes.py                ← All node functions (parse, critics,
│   │                              detect_disagreement, adjudicate, synthesize)
│   └── adjudicator.py          ← FinalAdjudicator class + AdjudicationResult schema
│
├── utils/
│   └── llm_clients.py          ← Instructor-patched client factories
│                                  (Gemini, Groq, OpenRouter, Ollama)
│
├── storage/
│   └── database.py             ← Async SQLite layer (aiosqlite)
│
├── api/
│   ├── app.py                  ← FastAPI app factory: lifespan, CORS, error handlers
│   ├── routes.py               ← All endpoint handlers
│   ├── models.py               ← Wire-format Pydantic models (request/response)
│   └── analytics.py            ← Aggregates stored verdicts into stats
│
└── ui/
    ├── app.py                  ← Streamlit entry point + sidebar
    ├── styles.py               ← CSS tokens, color palette, HTML helpers
    ├── verdict_view.py         ← Annotated response + issue expanders
    ├── critic_panel.py         ← Radar chart + five-critic comparison
    └── batch_view.py           ← Batch submission form + results table
```

---

## ⚙️ Setup and Running

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/download) installed and running locally
- Free API keys from: [Google AI Studio](https://aistudio.google.com/app/apikey) · [Groq](https://console.groq.com/keys) · [OpenRouter](https://openrouter.ai/keys)

### 1. Install

```bash
git clone <your-repo-url>
cd "LLM Output Arbitration System"

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

### 2. Pull local models

```bash
ollama pull gemma3
```

### 3. Configure keys

Fill in your `.env` file:

```env
GOOGLE_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
OPENROUTER_API_KEY=your_key_here
```

### 4. Run

```bash
# Run the Streamlit UI
streamlit run ui/app.py

# Run the REST API
uvicorn api.app:app --reload --port 8000
```

| Service | URL |
|---------|-----|
| Streamlit dashboard | http://localhost:8501 |
| REST API (Swagger docs) | http://localhost:8000/docs |
| Health check | http://localhost:8000/health |



---

## 🔌 Using the API

The REST API is fully documented at `http://localhost:8000/docs` (Swagger UI auto-generated by FastAPI).

### POST /v1/arbitrate — evaluate a single response

```bash
curl -X POST http://localhost:8000/v1/arbitrate \
  -H "Content-Type: application/json" \
  -d '{
    "original_prompt": "Explain how HTTPS works, including the TLS handshake.",
    "llm_response": "HTTPS uses SSL to encrypt traffic. The handshake was invented by Google in 2010..."
  }'
```

**Response (condensed):**
```json
{
  "recommendation": "REVISE",
  "overall_score": 68,
  "quality_score": 7,
  "overall_confidence": 0.81,
  "critic_agreement": 0.94,
  "executive_summary": "The response provides a broadly accurate overview of HTTPS but contains a critical factual error — TLS was not invented by Google. The asymmetric/symmetric description is reversed. Recommend revision.",
  "key_issues": [
    {
      "quote": "invented by Google in 2010",
      "explanation": "TLS evolved from Netscape's SSL and was standardised by the IETF, not invented by Google.",
      "severity": "major",
      "recommendation": "Replace with: 'TLS was standardised by the IETF, evolving from Netscape's SSL protocol.'"
    }
  ],
  "dimension_scores": {
    "accuracy": 55,
    "logic": 80,
    "completeness": 65,
    "safety": 95,
    "style": 82
  },
  "adjudication_used": true,
  "fast_path": false,
  "total_latency_ms": 18420
}
```

### POST /v1/arbitrate/batch — evaluate up to 10 responses

```bash
curl -X POST http://localhost:8000/v1/arbitrate/batch \
  -H "Content-Type: application/json" \
  -d '{
    "original_prompt": "What is 2 + 2?",
    "responses": ["4", "It is approximately 4.", "22", "The answer depends on context."]
  }'
```

### GET /v1/arbitrations/{id} — retrieve a stored verdict

```bash
curl http://localhost:8000/v1/arbitrations/550e8400-e29b-41d4-a716-446655440000
```

### GET /v1/analytics — aggregated system stats

```bash
curl http://localhost:8000/v1/analytics
```

Returns per-critic pass rates, average scores, overrule rates (how often the adjudicator changed a critic's score), fallback rates, and latency percentiles.

---

## 🖥️ Streamlit Dashboard

Launch with `streamlit run ui/app.py` → open http://localhost:8501.

### Sidebar

Enter any prompt and LLM response, then click **▶ Run**. The pipeline runs in the background and results appear automatically. Click **Sample** to load a pre-built demo with intentional errors (wrong facts, reversed encryption explanation, overconfident security claim).

### Verdict Explorer

The main evaluation view:

- **Score cards** — Overall score (0–100), Quality score (1–10), Confidence %, Critic Agreement %, Recommendation badge
- **Executive summary** — The adjudicator's official one-paragraph report
- **Annotated response** — The original LLM response with colour-coded inline highlights:

  | Colour | Severity |
  |--------|----------|
  | 🔴 Red underline | CRITICAL issue |
  | 🟠 Orange underline | MAJOR issue |
  | 🟡 Yellow underline | MINOR issue |
  | 🟢 Green underline | Identified strength |

  Click any highlight to see which critic flagged it, the evidence, and the fix recommendation.

- **Issue details** — Expandable cards for each confirmed issue with full evidence and adjudicator notes
- **Strengths** — Specific positive qualities confirmed by the adjudicator
- **Dismissed issues** — Issues one critic raised that the adjudicator ruled were false positives, with reasoning

### Critic Comparison

- **Radar chart** — Five-axis polar chart showing each critic's score
- **Per-critic cards** — Score, grade (A–F), confidence, pass/fail badge, and all issues for each of the five critics
- **Agreement matrix** — A 5×5 grid showing where critics agreed vs disagreed (colour-coded)
- **Outlier banner** — Highlights if one critic's score is a statistical outlier from the others

### Batch Mode

- Submit up to 10 LLM responses for the same prompt simultaneously
- Progress bar showing per-item status
- Sortable results table (by score, confidence, recommendation, issue count)
- Summary statistics: average score, approve/revise/reject breakdown, total latency

---

## 🛠️ Configuration Reference

All settings are in `.env` and loaded into a typed `Settings` object via pydantic-settings. Every value can be overridden without touching any Python code.

```env
# ── API Keys ──────────────────────────────────────────
GOOGLE_API_KEY=...
GROQ_API_KEY=...
OPENROUTER_API_KEY=...
OLLAMA_BASE_URL=http://localhost:11434

# ── Model names ────────────────────────────────────────
ACCURACY_MODEL=gemini-2.5-flash
LOGIC_MODEL=llama-3.3-70b-versatile
COMPLETENESS_MODEL=gemma3
SAFETY_MODEL=qwen/qwen3-235b-a22b:free
STYLE_MODEL=gemma3
ADJUDICATOR_MODEL=gemini-2.5-flash

# ── Fallback models ────────────────────────────────────
ACCURACY_FALLBACK_MODEL=llama-3.3-70b-versatile
ACCURACY_FALLBACK_PROVIDER=groq
LOGIC_FALLBACK_MODEL=llama-3.1-8b-instant
LOGIC_FALLBACK_PROVIDER=groq
SAFETY_FALLBACK_MODEL=gemma3
SAFETY_FALLBACK_PROVIDER=ollama

# ── Orchestration thresholds ───────────────────────────
FAST_PATH_THRESHOLD=80           # All scores >= this + all passed → skip adjudicator
DISAGREEMENT_VARIANCE_THRESHOLD=15.0  # Score std-dev above which adjudicator is called
FALLBACK_CONFIDENCE_PENALTY=0.05 # Confidence deducted per fallback model used

# ── Timeouts ───────────────────────────────────────────
CRITIC_TIMEOUT_SECONDS=30        # Remote API timeout before switching to fallback
OLLAMA_TIMEOUT_SECONDS=120       # Local model timeout (cold-start is slow)
MAX_CONCURRENT_CRITICS=5
```

---

## 💰 Cost

All providers used have free tiers. Running this project costs nothing.

| Provider | Model | Tier |
|----------|-------|------|
| Google AI Studio | Gemini 2.5 Flash | Free (1M tokens/day) |
| Groq | LLaMA 3.3 70B Versatile | Free (daily quota) |
| OpenRouter | Qwen3 235B (`:free`) | Free (rate-limited) |
| Ollama | Gemma 3 | Local — fully free, unlimited |

---

## 🧰 Tech Stack

| Technology | Role |
|-----------|------|
| **LangGraph** | DAG orchestration, parallel fan-out via `Send` API, typed state management |
| **Pydantic v2** | All data models, validation, enum normalisation |
| **Instructor** | Forces LLMs to return valid structured JSON matching the Pydantic schema |
| **FastAPI** | Async REST API, auto-generated OpenAPI/Swagger docs |
| **Streamlit** | Interactive dashboard — no frontend build step |
| **aiosqlite** | Async SQLite persistence layer for all verdicts |
| **pydantic-settings** | Typed `.env` loading with validation |
| **Tenacity** | Retry logic with exponential backoff on LLM API failures |
| **Ollama** | Runs Gemma 3 locally — no GPU required |

---

## 🚀 Future Work

- [ ] Streaming — surface critic results in the UI as they arrive instead of waiting for all five
- [ ] Custom critic weights — let users prioritise safety over style for certain use cases
- [ ] PDF / Markdown verdict export
- [ ] Critic confidence calibration using ground-truth labelled datasets
- [ ] Multimodal support — evaluate LLM responses that include images
- [ ] Webhook support for async evaluation in production pipelines
- [ ] Support for additional providers (Anthropic, Mistral, Cohere)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built as a final-year computer science project exploring multi-agent LLM evaluation, structured output enforcement with Instructor + Pydantic, and parallel DAG orchestration with LangGraph.*
