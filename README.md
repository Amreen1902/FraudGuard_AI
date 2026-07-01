# 🛡️ FraudGuard AI — Multi-Agent Fraud Detection System

> **Kaggle AI Agents Capstone 2024 | Agents for Business Track**

A production-ready **multi-agent AI system** that detects credit card fraud in real-time using a coordinated pipeline of specialized agents: Feature Engineering → ML Scoring → LLM Explanation → Decision Aggregation → Alerting, all orchestrated by a central agent and backed by an MCP tool server.

---

## 🎯 Problem Statement

Credit card fraud costs businesses **$32 billion annually**. Traditional rule-based systems suffer from high false positive rates and provide no reasoning behind decisions. FraudGuard AI addresses this by combining:

- **XGBoost ML model** (ROC-AUC: 0.9993) for accurate fraud scoring
- **LLM-powered explanations** (Claude API) for human-readable reasoning
- **Multi-agent coordination** for modular, scalable decision-making
- **MCP Server** for persistent audit logging and historical query tools

---

## 🏗️ Architecture

```
Transaction Input (JSON / CSV)
         │
         ▼
┌─────────────────────┐
│  Orchestrator Agent │  ← Coordinates the entire pipeline
└────────┬────────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌──────────┐  ┌──────────────┐
│ Feature  │  │  ML Scorer   │
│  Agent   │→ │    Agent     │  ← XGBoost (ROC-AUC 0.9993)
└──────────┘  └──────┬───────┘
                     │
              ┌──────▼────────┐
              │  Explanation  │
              │    Agent      │  ← Claude API (LLM reasoning)
              └──────┬────────┘
                     │
              ┌──────▼────────┐
              │   Aggregator  │
              │    Agent      │  ← Combines all outputs
              └──────┬────────┘
                     │
              ┌──────▼────────┐
              │  Alert Agent  │  ← Issues verdict, logs to audit
              └──────┬────────┘
                     │
              ┌──────▼────────┐
              │  MCP Server   │  ← DB queries, audit tools, stats
              └───────────────┘
```

### Agent Roles

| Agent | Responsibility |
|---|---|
| **OrchestratorAgent** | Routes transactions, manages pipeline flow, exposes MCP tools |
| **FeatureAgent** | Validates input, engineers 33 features (log-amount, hour, night flag, V-ratios) |
| **ScorerAgent** | Runs XGBoost inference, returns fraud probability + risk tier |
| **ExplanationAgent** | Calls Claude API to generate plain-English fraud reasoning |
| **AggregatorAgent** | Combines ML score + explanation into a structured audit record |
| **AlertAgent** | Issues final verdict, writes to append-only JSONL audit log |
| **MCPServer** | Exposes 4 tools: `query_audit_log`, `get_transaction`, `get_stats`, `flag_for_review` |

---

## ✅ Kaggle Evaluation Checklist

| Concept | Implementation |
|---|---|
| ✅ Multi-agent system (ADK) | 6 specialized agents + Orchestrator |
| ✅ MCP Server | `tools/mcp_server.py` — 4 audit/query tools |
| ✅ Deployability | Streamlit dashboard (`streamlit run dashboard/app.py`) |
| ✅ LLM Integration | Claude API in `agents/explanation_agent.py` |
| ✅ Security | No API keys in code; `.env` only; audit log immutable |

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/Amreen1902/FraudGuard_AI
cd fraud-detection-agent
pip install -r requirements.txt
```

### 2. Set API Key (optional — LLM explanations)

```bash
cp .env.example .env
# Edit .env and add: ANTHROPIC_API_KEY=your_key_here
```

> Without the API key, the system uses rule-based explanations automatically.

### 3. Generate Data & Train Model

```bash
python data/generate_data.py   # Creates data/transactions.csv
python models/train.py          # Trains XGBoost, saves to models/
```

### 4. Run the Agent

```bash
# Demo mode (3 sample transactions)
python main.py

# Batch mode (20 transactions)
python main.py --batch 20

# Show MCP audit stats
python main.py --stats
```

### 5. Launch Dashboard

```bash
streamlit run dashboard/app.py
```

Open http://localhost:8501 in your browser.

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| ROC-AUC | **0.9993** |
| Fraud Precision | 0.91 |
| Fraud Recall | 0.96 |
| F1 Score | 0.94 |
| Overall Accuracy | 0.99 |

---

## 📁 Project Structure

```
fraud-detection-agent/
├── agents/
│   ├── base_agent.py          # Abstract base with logging + error handling
│   ├── feature_agent.py       # Feature engineering (33 features)
│   ├── scorer_agent.py        # XGBoost inference + risk tier
│   ├── explanation_agent.py   # Claude API / rule-based fallback
│   ├── aggregator_agent.py    # Combines outputs into audit record
│   ├── alert_agent.py         # Issues verdict, writes audit log
│   └── orchestrator_agent.py  # Pipeline coordinator
├── tools/
│   └── mcp_server.py          # MCP tool server (4 audit tools)
├── models/
│   ├── train.py               # XGBoost training script
│   ├── fraud_model.pkl        # Trained model (generated)
│   ├── scaler.pkl             # StandardScaler (generated)
│   └── feature_names.pkl      # Feature list (generated)
├── data/
│   ├── generate_data.py       # Synthetic dataset generator
│   ├── transactions.csv       # Dataset (generated)
│   └── audit_log.jsonl        # Append-only audit trail (generated)
├── dashboard/
│   └── app.py                 # Streamlit 4-page dashboard
├── tests/
│   └── test_agents.py         # 20 unit + integration tests
├── main.py                    # CLI entry point
├── requirements.txt
└── README.md
```

---

## 🔒 Security Features

- **No API keys in source code** — loaded via `python-dotenv` from `.env`
- **Audit log is append-only** — decisions are immutable for compliance
- **Input validation** — FeatureAgent checks all required fields before processing
- **Error isolation** — each agent wrapped in try/catch; pipeline fails gracefully

---

## 🧪 Running Tests

```bash
python -m pytest tests/ -v
# 20 tests — all pass ✅
```

---

## 🛠️ Technologies Used

- **Python 3.12**
- **XGBoost** — fraud classification model
- **Scikit-learn** — preprocessing, evaluation
- **Anthropic Claude API** — LLM-powered explanations
- **Streamlit + Plotly** — interactive dashboard
- **Pandas / NumPy** — data processing
- **pytest** — testing

---

## 📹 Video Demo

[YouTube Link — add after recording]

---

## 👩‍💻 Author

Built for the **Kaggle AI Agents Capstone 2024** — Agents for Business Track.
