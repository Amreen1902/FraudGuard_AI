"""
Fraud Detection Dashboard — Streamlit App
Run: streamlit run dashboard/app.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from agents.orchestrator_agent import OrchestratorAgent
from tools.mcp_server import MCPServer

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .metric-card {
    background: linear-gradient(135deg, #1e1e2e, #2a2a3e);
    border: 1px solid #3a3a5c;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    color: white;
  }
  .metric-value { font-size: 2.2rem; font-weight: 700; }
  .metric-label { font-size: 0.85rem; color: #aaa; margin-top: 4px; }
  .verdict-BLOCK  { color: #ff4b4b; font-weight: 700; }
  .verdict-HOLD   { color: #ff9800; font-weight: 700; }
  .verdict-REVIEW { color: #ffd700; font-weight: 700; }
  .verdict-APPROVE{ color: #00c853; font-weight: 700; }
  .agent-log {
    background: #0e1117;
    border-radius: 8px;
    padding: 12px;
    font-family: monospace;
    font-size: 0.78rem;
    max-height: 280px;
    overflow-y: auto;
    border: 1px solid #2a2a3e;
  }
</style>
""", unsafe_allow_html=True)

# ── Cached resources ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_orchestrator():
    return OrchestratorAgent()

@st.cache_resource(show_spinner=False)
def load_mcp():
    return MCPServer()

orchestrator = load_orchestrator()
mcp          = load_mcp()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ FraudGuard AI")
    st.caption("Multi-Agent Fraud Detection System")
    st.divider()
    page = st.radio(
        "Navigate",
        ["📊 Dashboard", "🔍 Analyze Transaction", "📁 Batch Analysis", "🗄️ Audit Log"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("**Agent Status**")
    agents = ["OrchestratorAgent","FeatureAgent","ScorerAgent",
              "ExplanationAgent","AggregatorAgent","AlertAgent","MCPServer"]
    for a in agents:
        st.markdown(f"🟢 `{a}`")
    st.divider()
    st.caption("Kaggle AI Agents Capstone 2024\nAgents for Business Track")

# ── Helper: load audit records ─────────────────────────────────────────────────
def load_audit() -> pd.DataFrame:
    path = "data/audit_log.jsonl"
    if not os.path.exists(path):
        return pd.DataFrame()
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except:
                    pass
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df["amount"] = df["raw_transaction"].apply(lambda x: x.get("Amount", 0) if isinstance(x, dict) else 0)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — Dashboard
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.title("📊 Fraud Detection Dashboard")
    df = load_audit()

    if df.empty:
        st.info("No transactions analyzed yet. Use **Analyze Transaction** or **Batch Analysis** to get started.")
    else:
        # ── KPI row ────────────────────────────────────────────────────────
        total  = len(df)
        fraud  = len(df[df["final_verdict"].isin(["BLOCK","HOLD"])])
        rate   = fraud / total * 100
        avg_p  = df["fraud_probability"].mean() * 100
        blocked= len(df[df["final_verdict"] == "BLOCK"])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Transactions", f"{total:,}")
        c2.metric("Fraud Detected",     f"{fraud:,}", delta=f"{rate:.1f}% rate", delta_color="inverse")
        c3.metric("Blocked",            f"{blocked:,}")
        c4.metric("Avg Fraud Prob",     f"{avg_p:.1f}%")

        st.divider()
        col1, col2 = st.columns(2)

        # ── Verdict pie ────────────────────────────────────────────────────
        with col1:
            st.subheader("Verdict Distribution")
            verdict_counts = df["final_verdict"].value_counts().reset_index()
            verdict_counts.columns = ["Verdict", "Count"]
            color_map = {"APPROVE":"#00c853","REVIEW":"#ffd700","HOLD":"#ff9800","BLOCK":"#ff4b4b"}
            fig = px.pie(
                verdict_counts, names="Verdict", values="Count",
                color="Verdict", color_discrete_map=color_map,
                hole=0.45,
            )
            fig.update_layout(margin=dict(t=20,b=20,l=20,r=20), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

        # ── Risk tier bar ──────────────────────────────────────────────────
        with col2:
            st.subheader("Risk Tier Breakdown")
            tier_counts = df["risk_tier"].value_counts().reindex(
                ["LOW","MEDIUM","HIGH","CRITICAL"], fill_value=0
            ).reset_index()
            tier_counts.columns = ["Tier","Count"]
            tier_colors = {"LOW":"#00c853","MEDIUM":"#ffd700","HIGH":"#ff9800","CRITICAL":"#ff4b4b"}
            fig2 = px.bar(
                tier_counts, x="Tier", y="Count",
                color="Tier", color_discrete_map=tier_colors,
                text="Count",
            )
            fig2.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=20,b=20))
            fig2.update_traces(textposition="outside")
            st.plotly_chart(fig2, use_container_width=True)

        # ── Fraud probability histogram ────────────────────────────────────
        st.subheader("Fraud Probability Distribution")
        fig3 = px.histogram(
            df, x="fraud_probability", nbins=30, color_discrete_sequence=["#7c5cbf"],
            labels={"fraud_probability": "Fraud Probability"},
        )
        fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           margin=dict(t=20,b=20))
        st.plotly_chart(fig3, use_container_width=True)

        # ── Recent HIGH/CRITICAL table ─────────────────────────────────────
        st.subheader("🔴 Recent High-Risk Transactions")
        hi = df[df["final_verdict"].isin(["BLOCK","HOLD"])].sort_values(
            "timestamp", ascending=False
        ).head(10)[["transaction_id","timestamp","amount","fraud_probability","risk_tier","final_verdict"]]
        hi.columns = ["TXN ID","Timestamp","Amount","Fraud Prob","Tier","Verdict"]
        hi["Fraud Prob"] = hi["Fraud Prob"].apply(lambda x: f"{x:.1%}")
        hi["Amount"] = hi["Amount"].apply(lambda x: f"${x:.2f}")
        st.dataframe(hi, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — Analyze Single Transaction
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Analyze Transaction":
    st.title("🔍 Analyze a Transaction")
    st.caption("Enter transaction details below and run the full multi-agent pipeline.")

    col1, col2 = st.columns(2)
    with col1:
        amount = st.number_input("Amount ($)", min_value=0.01, max_value=50000.0, value=1249.99, step=0.01)
        time_val = st.slider("Time (seconds since epoch)", 0, 172800, 82800)
    with col2:
        preset = st.selectbox("Load a preset", [
            "Custom",
            "🔴 High-Risk (night, high amount)",
            "✅ Normal (daytime, small amount)",
            "🟡 Medium Risk",
        ])

    PRESETS = {
        "🔴 High-Risk (night, high amount)": {
            "Time":82800,"Amount":1249.99,"V1":-3.8,"V2":1.9,"V3":-1.1,"V4":2.3,
            "V5":-0.5,"V6":-0.8,"V7":0.4,"V8":0.1,"V9":-2.2,"V10":-0.3,
            "V11":1.1,"V12":-1.9,"V13":0.2,"V14":-2.1,"V15":0.5,"V16":-0.7,
            "V17":-0.9,"V18":0.3,"V19":-0.1,"V20":0.6,"V21":-0.4,"V22":0.2,
            "V23":-0.3,"V24":0.1,"V25":-0.5,"V26":0.4,"V27":-0.2,"V28":0.1,
        },
        "✅ Normal (daytime, small amount)": {
            "Time":36000,"Amount":45.50,"V1":0.2,"V2":-0.3,"V3":0.4,"V4":-0.1,
            "V5":0.3,"V6":0.1,"V7":-0.2,"V8":0.0,"V9":0.1,"V10":0.2,
            "V11":-0.1,"V12":0.3,"V13":-0.2,"V14":0.1,"V15":0.0,"V16":-0.1,
            "V17":0.2,"V18":-0.3,"V19":0.1,"V20":0.0,"V21":-0.1,"V22":0.2,
            "V23":0.0,"V24":-0.1,"V25":0.1,"V26":0.0,"V27":-0.2,"V28":0.0,
        },
        "🟡 Medium Risk": {
            "Time":54000,"Amount":380.00,"V1":-1.5,"V2":0.8,"V3":-0.5,"V4":1.2,
            "V5":-0.2,"V6":-0.3,"V7":0.1,"V8":0.2,"V9":-0.9,"V10":-0.1,
            "V11":0.5,"V12":-0.8,"V13":0.1,"V14":-0.9,"V15":0.2,"V16":-0.3,
            "V17":-0.4,"V18":0.1,"V19":0.0,"V20":0.3,"V21":-0.2,"V22":0.1,
            "V23":-0.1,"V24":0.0,"V25":-0.2,"V26":0.1,"V27":-0.1,"V28":0.0,
        },
    }

    if preset != "Custom":
        txn_data = PRESETS[preset].copy()
    else:
        txn_data = {"Time": time_val, "Amount": amount}
        for i in range(1, 29):
            txn_data[f"V{i}"] = 0.0

    if st.button("🚀 Run Agent Pipeline", type="primary", use_container_width=True):
        with st.spinner("Running multi-agent pipeline..."):
            result = orchestrator.analyze(txn_data)

        verdict = result.get("verdict","ERROR")
        tier    = result.get("tier","UNKNOWN")
        prob    = result.get("probability", 0) or 0
        emoji   = result.get("emoji","⚪")

        color_map = {"APPROVE":"green","REVIEW":"orange","HOLD":"orange","BLOCK":"red","ERROR":"gray"}
        st.markdown(f"### {emoji} Verdict: :{color_map.get(verdict,'gray')}[{verdict}]")

        m1, m2, m3 = st.columns(3)
        m1.metric("Fraud Probability", f"{prob:.1%}")
        m2.metric("Risk Tier", tier)
        m3.metric("Transaction ID", result.get("transaction_id","—"))

        st.subheader("📝 Explanation")
        st.info(result.get("explanation","No explanation available."))

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("⚠️ Risk Signals")
            for s in result.get("risk_signals", []):
                st.markdown(f"- {s}")
        with col_b:
            st.subheader("📈 Top Features")
            feats = result.get("top_features", [])
            if feats:
                feat_df = pd.DataFrame(feats)
                fig = px.bar(feat_df, x="importance", y="feature", orientation="h",
                             color_discrete_sequence=["#7c5cbf"])
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=10,b=10))
                st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — Batch Analysis
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📁 Batch Analysis":
    st.title("📁 Batch Transaction Analysis")
    n = st.slider("Number of transactions to analyze", 5, 50, 20)

    if st.button("▶ Run Batch Analysis", type="primary", use_container_width=True):
        progress = st.progress(0, text="Starting pipeline...")
        with st.spinner(f"Analyzing {n} transactions..."):
            results = orchestrator.analyze_batch("data/transactions.csv", n_samples=n)
            progress.progress(100, text="Done!")

        df_results = pd.DataFrame([{
            "TXN ID":    r.get("transaction_id",""),
            "Verdict":   r.get("verdict",""),
            "Tier":      r.get("tier",""),
            "Prob":      f"{(r.get('probability') or 0):.1%}",
            "Emoji":     r.get("emoji",""),
        } for r in results])

        st.subheader("Results")
        st.dataframe(df_results, use_container_width=True, hide_index=True)

        # Summary chart
        vc = df_results["Verdict"].value_counts().reset_index()
        vc.columns = ["Verdict","Count"]
        color_map = {"APPROVE":"#00c853","REVIEW":"#ffd700","HOLD":"#ff9800","BLOCK":"#ff4b4b"}
        fig = px.bar(vc, x="Verdict", y="Count", color="Verdict",
                     color_discrete_map=color_map, text="Count")
        fig.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        st.success(f"Batch complete! {len(results)} transactions processed. Audit log updated.")

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — Audit Log
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗄️ Audit Log":
    st.title("🗄️ Audit Log (MCP Server)")
    st.caption("All decisions are persisted by the AlertAgent and queryable via the MCP Server tool.")

    df = load_audit()
    if df.empty:
        st.info("Audit log is empty. Run some transactions first.")
    else:
        # MCP Stats
        stats = mcp.call("get_stats")
        s1,s2,s3,s4 = st.columns(4)
        s1.metric("Total Records",  stats.get("total_transactions",0))
        s2.metric("Fraud Rate",     f"{stats.get('fraud_rate',0):.1%}")
        s3.metric("Avg Fraud Prob", f"{stats.get('avg_fraud_probability',0):.1%}")
        s4.metric("Blocked",        stats.get("by_verdict",{}).get("BLOCK",0))

        st.divider()

        # Filters
        col1, col2 = st.columns(2)
        with col1:
            f_verdict = st.multiselect("Filter by Verdict",
                ["APPROVE","REVIEW","HOLD","BLOCK"], default=["APPROVE","REVIEW","HOLD","BLOCK"])
        with col2:
            f_tier = st.multiselect("Filter by Tier",
                ["LOW","MEDIUM","HIGH","CRITICAL"], default=["LOW","MEDIUM","HIGH","CRITICAL"])

        mask = df["final_verdict"].isin(f_verdict) & df["risk_tier"].isin(f_tier)
        filtered = df[mask].sort_values("timestamp", ascending=False)

        display = filtered[["transaction_id","timestamp","amount","fraud_probability","risk_tier","final_verdict","confidence"]].copy()
        display.columns = ["TXN ID","Timestamp","Amount","Fraud Prob","Tier","Verdict","Confidence"]
        display["Fraud Prob"]  = display["Fraud Prob"].apply(lambda x: f"{x:.1%}")
        display["Confidence"]  = display["Confidence"].apply(lambda x: f"{x:.1%}")
        display["Amount"]      = display["Amount"].apply(lambda x: f"${x:.2f}")
        st.dataframe(display, use_container_width=True, hide_index=True)

        st.caption(f"Showing {len(filtered)} of {len(df)} records")
