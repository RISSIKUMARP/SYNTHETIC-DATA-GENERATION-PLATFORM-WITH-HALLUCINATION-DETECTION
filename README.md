# Synthetic Data Generation Platform with AI Hallucination Detection

A privacy-preserving system that generates realistic synthetic banking transactions using CTGAN and validates them through a multi-agent AI pipeline with hierarchical hallucination detection — catching bad outputs from both the generative model and the LLM validator itself.

Built on the [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset (284,807 transactions, 0.17% fraud rate). Generates 100K synthetic transactions that are statistically identical to real data with zero PII leakage.

---

## Why This Exists

Real transaction data is sensitive, regulated, and hard to share. Synthetic data solves this — but only if it's actually good. Bad synthetic data (hallucinations) can look plausible while being statistically impossible.

This project tackles both problems: generate high-fidelity fake data, then deploy an autonomous AI system to catch anything the generator got wrong.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   ORCHESTRATOR AGENT                     │
│         Pipeline manager · Pass/fail decisions           │
│         Auto-triggers CTGAN re-generation if needed      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  CTGAN        │  │  Privacy     │  │  Quality     │  │
│  │  Generator    │──│  Layer       │──│  Report      │  │
│  │  (100K rows)  │  │  (k-anon,   │  │  (PDF)       │  │
│  │              │  │  diff priv)  │  │              │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                          │
│  ┌────────────────── VALIDATION AGENTS ────────────────┐ │
│  │                                                      │ │
│  │  Agent 1: Rule Validator                             │ │
│  │  Business logic + LLM-generated rule discovery       │ │
│  │                                                      │ │
│  │  Agent 2: Statistical Validator                      │ │
│  │  KS tests, correlations, autonomous failure drill-   │ │
│  │  down and diagnosis                                  │ │
│  │                                                      │ │
│  │  Agent 3: LLM Semantic Validator (GPT-4o-mini)       │ │
│  │  Natural language reasoning on flagged records        │ │
│  │  Can request context from Agent 2                    │ │
│  │                                                      │ │
│  │  Agent 4: RAG Similarity Validator (Pinecone)        │ │
│  │  Embedding search for nearest real neighbors         │ │
│  │  Escalates to Agent 3 when no close match found      │ │
│  │                                                      │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                          │
│  ┌──────────────────────────────────────────────────────┐ │
│  │  META-VALIDATION: LLM Hallucination Check            │ │
│  │  Verifies GPT-4o-mini's own reasoning is grounded    │ │
│  │  in actual data (catches validator hallucinations)    │ │
│  └──────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

**Two-level hallucination detection:**
- **Level 1** — CTGAN output: catches impossible synthetic transactions
- **Level 2** — LLM validator: catches when GPT-4o-mini hallucinates during its own validation reasoning (e.g., claims "Amount is negative" when it's actually 50.00)

---

## Tech Stack

| Category | Tools |
|---|---|
| **Generation** | Python, SDV/CTGAN, PyTorch |
| **GenAI/LLM** | OpenAI API (GPT-4o-mini), text-embedding-3-small |
| **RAG** | Pinecone vector database |
| **Validation** | SciPy (KS tests), Scikit-learn (outlier detection) |
| **API** | FastAPI, Uvicorn |
| **Dashboard** | Streamlit |
| **Deployment** | Docker, Docker Compose |

---

## Project Structure

```
synthetic-data-project/
├── data/
│   ├── raw/                  # Kaggle dataset (not committed)
│   └── synthetic/            # Generated synthetic data
├── scripts/
│   ├── verify_setup.py       # Environment verification
│   └── eda_baseline.py       # EDA + baseline statistics
├── src/
│   ├── generation/           # CTGAN pipeline
│   ├── validation/           # Agentic validation system
│   │   ├── orchestrator.py
│   │   ├── rule_agent.py
│   │   ├── statistical_agent.py
│   │   ├── llm_agent.py
│   │   └── rag_agent.py
│   ├── privacy/              # Differential privacy + k-anonymity
│   ├── api/                  # FastAPI endpoints
│   └── dashboard/            # Streamlit UI
├── reports/
│   ├── eda/                  # EDA visualizations + baseline JSON
│   └── quality/              # Validation reports
├── tests/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.template
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- pip
- ~2 GB disk space (PyTorch + SDV are large)

### Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/synthetic-data-project.git
cd synthetic-data-project

# Create virtual environment
python -m venv venv

# Activate it
# Mac/Linux:
source venv/bin/activate
# Windows PowerShell:
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Download the Dataset

1. Go to [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Download `creditcard.csv`
3. Place it in `data/raw/creditcard.csv`

### Verify Installation

```bash
python scripts/verify_setup.py
```

All checks should pass, including a CTGAN smoke test that trains on 500 rows and generates 5 synthetic records.

### API Keys (needed for validation agents)

Copy `.env.template` to `.env` and add your keys:

```bash
cp .env.template .env
```

```
OPENAI_API_KEY=your-key-here
PINECONE_API_KEY=your-key-here
PINECONE_ENVIRONMENT=your-environment
```

---

## Dataset

**Kaggle Credit Card Fraud Detection** — 284,807 European credit card transactions from September 2013.

| Column | Description |
|---|---|
| `Time` | Seconds elapsed from first transaction |
| `V1`–`V28` | PCA-transformed features (anonymized) |
| `Amount` | Transaction amount |
| `Class` | 0 = legitimate, 1 = fraud (0.17% fraud rate) |

The extreme class imbalance (492 fraud out of 284,807) is a key challenge — synthetic generation must preserve this ratio, and the validation system must catch drift.

---

## How It Works

### 1. Generate
CTGAN learns the joint distribution of all 31 features from real data and generates 100K synthetic transactions. The privacy layer applies differential privacy and ensures k-anonymity (k=100).

### 2. Validate
Four autonomous agents, coordinated by an orchestrator:

- **Rule Agent** — enforces business constraints (amount ranges, feature bounds) and discovers new rules by prompting GPT-4o-mini to analyze the real dataset
- **Statistical Agent** — runs Kolmogorov-Smirnov tests, correlation comparisons, and distribution matching. Autonomously drills into failures to diagnose root causes
- **LLM Agent** — GPT-4o-mini reads flagged transactions and reasons about whether they make business sense. Can pull additional context from the Statistical Agent
- **RAG Agent** — embeds synthetic records using `text-embedding-3-small`, searches Pinecone for nearest real neighbors. No close match → escalates to LLM Agent

Agents communicate with each other. The orchestrator can loop, retry, and trigger re-generation without human intervention.

### 3. Meta-Validate
A separate check verifies that GPT-4o-mini's own validation reasoning is grounded in actual data — catching cases where the LLM hallucinates facts about the records it's reviewing.

---

## Success Metrics

| Metric | Target |
|---|---|
| Synthetic records generated | 100,000 |
| Statistical fidelity (vs real data) | ≥ 95% |
| Hallucination detection accuracy | ≥ 98% |
| PII leakage | 0% (k-anonymity k=100) |

---

## Running the Pipeline

```bash
# Generate synthetic data
python -m src.generation.pipeline

# Run validation agents
python -m src.validation.orchestrator

# Start the API
uvicorn src.api.main:app --reload

# Launch the dashboard
streamlit run src/dashboard/app.py
```

### Docker

```bash
docker-compose up --build
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/generate` | Generate synthetic records |
| `GET` | `/validate/{batch_id}` | Run validation on a batch |
| `GET` | `/report/{batch_id}` | Get quality report |
| `GET` | `/health` | Health check |

---

## License

MIT

---

## Acknowledgments

- Dataset: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) by ML Group, ULB
- Synthetic generation: [SDV (Synthetic Data Vault)](https://sdv.dev/)
