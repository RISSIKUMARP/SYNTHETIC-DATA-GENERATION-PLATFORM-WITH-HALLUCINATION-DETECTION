# Live Demo Guide

## Synthetic Data Generation Platform with AI Hallucination Detection

---

## Pre-Demo Setup (Do This Before the Demo)

### 1. Environment

```powershell
# Open terminal in project root
cd "D:\Local Disk D Files\Git files\SYNTHETIC-DATA-GENERATION-PLATFORM-WITH-HALLUCINATION-DETECTION"

# Activate virtual environment
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& ".\venv\Scripts\Activate.ps1"
```

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 3. API Keys (Optional but Recommended for Full Demo)

Copy `.env.template` to `.env` and fill in your keys:

```
GEMINI_API_KEY=your-gemini-key        # Enables live LLM validation (Layers 1 & 3)
PINECONE_API_KEY=your-pinecone-key    # Enables live RAG validation (Layer 4)
```

> **No API keys?** The platform runs fully functional in mock/fallback mode. All 4 validation layers still work using rule-based and sklearn-based backends. You can demo everything without keys.

### 4. Pre-Generate Demo Data

```powershell
python demo_data.py --n 500
```

This creates `data/synthetic/demo_synthetic.csv` and `data/synthetic/reference_500.csv` so Tab 1 loads instantly.

### 5. Quick Smoke Test

```powershell
python scripts/verify_setup.py
```

Confirms Python version, libraries, and config are all good.

---

## Launch the Demo

### Option A: Streamlit Dashboard (Recommended for Live Demo)

```powershell
streamlit run app.py
```

Opens at **http://localhost:8501**

### Option B: FastAPI REST API (For Technical Audiences)

```powershell
uvicorn api:app --reload --port 8000
```

Opens at **http://localhost:8000/docs** (Swagger UI)

### Option C: One-Command Launch (Linux/Mac/Git Bash)

```bash
bash run_demo.sh
```

Installs deps, generates data, and launches the dashboard automatically.

---

## Demo Walkthrough (5 Tabs)

### Tab 1: Data Generation (Start Here)

**What to show:**
1. Adjust the **sample size slider** (default 500, go up to 1000 for a richer demo)
2. Note the **fraud rate** input (0.173% -- matches real-world Kaggle dataset imbalance)
3. Click **"Generate Synthetic Data"**
4. Point out the metrics: total rows, fraud count, fraud rate
5. Show the **Real vs Synthetic fraud rate comparison chart** -- they should be nearly identical
6. Scroll down to the **data preview** -- 31 columns matching the Kaggle credit card schema

**Talking points:**
- Data is generated using distribution parameters derived from 284,807 real European credit card transactions
- If the real `creditcard.csv` is present in `data/raw/`, it bootstraps with noise to preserve inter-column correlations
- Without the real CSV, it falls back to hardcoded distribution parameters -- fully self-contained

---

### Tab 2: Validation Pipeline (Core Demo)

**What to show:**
1. Click **"Run Full Validation"**
2. Watch all 4 agents execute in sequence with progress indicators
3. Point out the **Agent Scorecard** -- 4 columns showing pass/fail, score (progress bar), and execution time
4. Show the **Score Breakdown Table**:

   | Layer | Weight | Score | Contribution |
   |-------|--------|-------|-------------|
   | Layer 1 (Rules) | 20% | ~0.95 | ~0.19 |
   | Layer 2 (Statistical) | 40% | ~0.85 | ~0.34 |
   | Layer 3 (LLM Semantic) | 25% | ~0.80 | ~0.20 |
   | Layer 4 (RAG Similarity) | 15% | ~0.90 | ~0.135 |
   | **Final** | **100%** | | **~0.87** |

5. If the score is below 0.75 (unlikely with good data), the **self-healing loop** triggers automatically -- show the retry log

**Talking points:**
- Multi-agent orchestration: each layer is an independent validation agent
- Agents communicate and escalate -- Layer 4 can trigger Layer 3 to re-examine flagged rows
- Weighted consensus scoring balances speed vs accuracy
- Self-healing: if validation fails, the system autonomously re-generates data with a different seed (up to 2 retries)

---

### Tab 3: Statistical Quality Analysis (Deep Dive)

**What to show:**
1. **KS Test Bar Chart** -- feature-by-feature distribution match:
   - Green bars (KS < 0.05): excellent match
   - Yellow bars (KS < 0.10): acceptable
   - Red bars (KS >= 0.10): poor match
2. Point out which features pass/fail -- Amount and Time often have larger KS statistics
3. **Fraud Rate Preservation** -- real vs synthetic fraud rate comparison
4. **Correlation Difference Heatmap** -- shows how well inter-feature relationships are preserved

**Talking points:**
- Kolmogorov-Smirnov test is a non-parametric test comparing two distributions
- Good synthetic data should have KS statistics close to 0 (identical distributions)
- Correlation preservation is critical -- synthetic data that passes KS tests but breaks correlations is still bad

---

### Tab 4: Hallucination Detection (Key Innovation)

**What to show:**

**Left Panel -- Level 1 (CTGAN Hallucinations):**
1. Rule violations count from Layer 1
2. RAG anomalies count from Layer 4
3. Top rule failures table (which rules were violated most)
4. RAG-flagged row indices

**Right Panel -- Level 2 (LLM Reasoning Hallucinations):**
1. Meta-validation pass/fail badge
2. Contradictions found (e.g., high score but many anomalies reported)
3. Hallucinations detected (e.g., LLM referenced non-existent columns like V99)
4. Confidence adjustment and adjusted Layer 3 score
5. Expand the LLM anomaly observations for detail

**Talking points:**
- **This is the key differentiator** -- two-level hallucination detection
- Level 1 catches when the CTGAN generates unrealistic data (impossible values, broken distributions)
- Level 2 catches when the LLM validator itself hallucinates (fabricates column names, contradicts its own reasoning)
- The meta-validator checks for: contradictions between score and findings, references to non-existent features, hedging language inconsistent with confidence scores

---

### Tab 5: Architecture (Reference)

**What to show:**
1. ASCII pipeline diagram -- full data flow from generation to decision
2. Technology stack table
3. Weighted scoring formula and self-healing explanation

**Talking points:**
- Use this tab to walk through the architecture at a high level
- Point out the escalation path: Layer 4 RAG can escalate to Layer 3 LLM for deeper analysis

---

## Bonus: API Demo (For Technical Audiences)

If you also launched the FastAPI server:

```powershell
# Health check
curl http://localhost:8000/health

# Generate synthetic data
curl -X POST http://localhost:8000/generate -H "Content-Type: application/json" -d "{\"n_samples\": 500}"

# Run full validation pipeline
curl -X POST http://localhost:8000/validate -H "Content-Type: application/json" -d "{\"n_samples\": 500}"

# Get last validation report
curl http://localhost:8000/report
```

Or use the interactive Swagger UI at **http://localhost:8000/docs**.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again |
| Streamlit won't start | Check port 8501 isn't in use: `netstat -ano \| findstr 8501` |
| "Config error" on validation | Copy `.env.template` to `.env` (keys are optional but file must exist) |
| Low KS scores / validation fails | Normal for distribution-generated data. With real `creditcard.csv` in `data/raw/`, bootstrap mode produces better results |
| Gemini/Pinecone errors | These are optional. Platform falls back to mock/sklearn mode automatically |
| `Set-ExecutionPolicy` error | Run PowerShell as Administrator, or use `cmd` and run `venv\Scripts\activate.bat` instead |

---

## Demo Script (Suggested Narrative)

> "This platform generates privacy-preserving synthetic banking transactions using CTGAN-derived distributions, then validates them through a multi-agent AI pipeline."

1. **[Tab 1]** "First, we generate 500 synthetic transactions matching the Kaggle credit card fraud dataset schema -- 31 features including PCA-transformed V-features, with realistic 0.17% fraud rate."

2. **[Tab 2]** "Now we run the full validation pipeline -- four independent AI agents each assess quality from a different angle: rule-based checks, statistical tests, LLM semantic reasoning, and RAG similarity matching. The system computes a weighted consensus score."

3. **[Tab 3]** "Here's the statistical deep dive -- KS tests show how closely each feature's synthetic distribution matches the real data. Green means excellent match."

4. **[Tab 4]** "This is our key innovation -- two-level hallucination detection. Level 1 catches when the generator produces unrealistic data. Level 2 catches when the LLM validator itself hallucinates -- referencing non-existent columns or contradicting its own findings."

5. **[Tab 5]** "The architecture shows the full pipeline with escalation paths and self-healing. If validation fails, the system autonomously re-generates with a different seed."

6. **[Tab 2, if applicable]** "And if we force a failure scenario, you can see the self-healing loop kick in -- the system retries with new seeds until it passes or exhausts retries."

---

## Key Numbers to Mention

- **284,807** real transactions in original Kaggle dataset
- **31** features (Time, V1-V28, Amount, Class)
- **0.173%** fraud rate (492 fraud out of 284,315 legitimate)
- **4** independent validation agents
- **5** total validation layers (including meta-validator)
- **0.75** pass threshold for weighted score
- **2** maximum self-healing retries
- Weights: **20%** rules, **40%** statistical, **25%** LLM, **15%** RAG
