<div align="center">

# 🛡️ FraudGuard AI

### Multi-Agent Fraud Detection System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-ROC--AUC%200.9993-brightgreen)](https://xgboost.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red?logo=streamlit)](https://streamlit.io)
[![Gemini](https://img.shields.io/badge/LLM-Gemini%201.5%20Flash-orange?logo=google)](https://aistudio.google.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

**Kaggle AI Agents Capstone 2024 — Agents for Business Track**

[Quick Start](#-quick-start) • [Architecture](#-architecture) • [Dashboard](#-dashboard)

</div>

---

## 📌 Problem Statement

Credit card fraud costs businesses **$32 billion annually**. Traditional rule-based systems have high false positive rates and provide **zero reasoning** behind their decisions — making it hard for analysts to act quickly.

**FraudGuard AI** solves this with a coordinated multi-agent system that:
- Detects fraud with **99.3% accuracy** (ROC-AUC: 0.9993)
- Explains *why* a transaction is suspicious in plain English (Gemini LLM)
- Logs every decision to an immutable audit trail via MCP Server
- Provides a real-time interactive dashboard for analysts

---

## 🏗️ Architecture

```
Transaction Input (JSON / CSV)
         │
         ▼
┌─────────────────────────┐
│    Orchestrator Agent   │  ← Central coordinator
└──────────┬──────────────┘
           │
     ┌─────┴──────┐
     ▼             ▼
┌──────────┐  ┌───────────────┐
│ Feature  │→ │  ML Scorer    │  ← XGBoost (ROC-AUC 0.9993)
│  Agent   │  │    Agent      │
└──────────┘  └──────┬────────┘
                     │
              ┌──────▼────────┐
              │  Explanation  │  ← Gemini 1.5 Flash (LLM)
              │    Agent      │
              └──────┬────────┘
                     │
              ┌──────▼────────┐
              │  Aggregator   │  ← Combines all outputs
              │    Agent      │
              └──────┬────────┘
                     │
              ┌──────▼────────┐
              │  Alert Agent  │  ← Final verdict + audit log
              └──────┬────────┘
                     │
              ┌──────▼────────┐
              │  MCP Server   │  ← 4 audit/query tools
              └───────────────┘
```

### Agent Roles

| Agent | Responsibility |
|---|---|
| **OrchestratorAgent** | Routes transactions, manages pipeline, exposes MCP tools |
| **FeatureAgent** | Validates input, engineers 33 features (log-amount, hour, night flag, V-ratios) |
| **ScorerAgent** | Runs XGBoost inference → fraud probability + risk tier |
| **ExplanationAgent** | Calls Gemini API → plain-English fraud reasoning |
| **AggregatorAgent** | Combines ML score + explanation → structured audit record |
| **AlertAgent** | Issues verdict (APPROVE/REVIEW/HOLD/BLOCK), writes to audit log |
| **MCPServer** | 4 tools: `query_audit_log`, `get_transaction`, `get_stats`, `flag_for_review` |

---

## ✅ Kaggle Evaluation Checklist

| Concept | Where | Status |
|---|---|---|
| Multi-agent system | `agents/orchestrator_agent.py` | ✅ 6 agents + Orchestrator |
| MCP Server | `tools/mcp_server.py` | ✅ 4 audit/query tools |
| LLM Integration | `agents/explanation_agent.py` | ✅ Gemini 1.5 Flash |
| Deployability | `dashboard/app.py` | ✅ Streamlit dashboard |
| Security features | `.env`, `.gitignore`, audit log | ✅ No keys in code |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Git

### 1. Clone the repo

```bash
git clone https://github.com/Amreen1902/FraudGuard_AI
cd fraud-detection-agent
```

### 2. Create virtual environment

**Mac / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up API Key (optional — for LLM explanations)

```bash
# Copy the example env file
cp .env.example .env
```

Edit `.env` and add your **Gemini API key** (free at [aistudio.google.com](https://aistudio.google.com)):
```
GEMINI_API_KEY=your_gemini_api_key_here
```

> ⚠️ Without the API key the system still works — rule-based explanations are used automatically.

### 5. Generate data & train model

```bash
python data/generate_data.py    # Creates data/transactions.csv (9,500 rows)
python models/train.py           # Trains XGBoost, saves to models/
```

### 6. Run the agent

```bash
# Demo mode — 3 sample transactions
python main.py

# Batch mode — analyze N transactions from CSV
python main.py --batch 20

# MCP audit stats
python main.py --stats
```

### 7. Launch dashboard

```bash
streamlit run dashboard/app.py
```

Open **http://localhost:8501** in your browser.

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| **ROC-AUC** | **0.9993** |
| Fraud Precision | 0.91 |
| Fraud Recall | 0.96 |
| F1 Score | 0.94 |
| Overall Accuracy | 0.99 |

Trained on 9,500 synthetic transactions (500 fraud, 9,000 legit) with 33 engineered features.

---

## 🖥️ Dashboard

4-page Streamlit dashboard:

| Page | Description |
|---|---|
| 📊 Dashboard | KPIs, verdict distribution, risk tier charts, high-risk table |
| 🔍 Analyze Transaction | Real-time single transaction analysis with presets |
| 📁 Batch Analysis | Analyze up to 50 transactions at once |
| 🗄️ Audit Log | MCP-powered query interface with filters |

---

## 📁 Project Structure

```
fraud-detection-agent/
├── agents/
│   ├── base_agent.py          # Abstract base — logging + error handling
│   ├── feature_agent.py       # 33 feature engineering
│   ├── scorer_agent.py        # XGBoost inference + risk tier
│   ├── explanation_agent.py   # Gemini API / rule-based fallback
│   ├── aggregator_agent.py    # Combines outputs → audit record
│   ├── alert_agent.py         # Final verdict + JSONL audit log
│   └── orchestrator_agent.py  # Pipeline coordinator
├── tools/
│   └── mcp_server.py          # MCP Server — 4 audit tools
├── models/
│   └── train.py               # XGBoost training script
├── data/
│   └── generate_data.py       # Synthetic dataset generator
├── dashboard/
│   └── app.py                 # Streamlit 4-page dashboard
├── tests/
│   └── test_agents.py         # 20 unit + integration tests
├── main.py                    # CLI entry point
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🔒 Security

- API keys loaded via `.env` — never hardcoded
- `.gitignore` excludes `.env`, model files, audit log
- Audit log is **append-only** for compliance
- Input validation in FeatureAgent before any processing
- Each agent isolated with try/catch — pipeline fails gracefully

---

## 🧪 Tests

```bash
python -m pytest tests/ -v
```

20 tests covering all agents — unit + end-to-end integration.

---

## 🛠️ Tech Stack

| Technology | Use |
|---|---|
| Python 3.10+ | Core language |
| XGBoost | Fraud classification model |
| Scikit-learn | Preprocessing + evaluation |
| Google Gemini 1.5 Flash | LLM explanations |
| Streamlit + Plotly | Interactive dashboard |
| Pandas / NumPy | Data processing |
| pytest | Testing |

---

## 👩‍💻 Author

Built for **Kaggle AI Agents Capstone 2024** — Agents for Business Track.
