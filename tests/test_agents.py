"""
tests/test_agents.py — Unit tests for all fraud detection agents.
Run: python -m pytest tests/ -v
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from agents.feature_agent import FeatureAgent
from agents.scorer_agent import ScorerAgent
from agents.explanation_agent import ExplanationAgent
from agents.aggregator_agent import AggregatorAgent
from agents.alert_agent import AlertAgent
from agents.orchestrator_agent import OrchestratorAgent
from tools.mcp_server import MCPServer

# ── Fixtures ───────────────────────────────────────────────────────────────────
SAMPLE_TXN = {
    "Time": 82800, "Amount": 1249.99,
    "V1": -3.8, "V2": 1.9, "V3": -1.1, "V4": 2.3,
    "V5": -0.5, "V6": -0.8, "V7": 0.4, "V8": 0.1,
    "V9": -2.2, "V10": -0.3, "V11": 1.1, "V12": -1.9,
    "V13": 0.2, "V14": -2.1, "V15": 0.5, "V16": -0.7,
    "V17": -0.9, "V18": 0.3, "V19": -0.1, "V20": 0.6,
    "V21": -0.4, "V22": 0.2, "V23": -0.3, "V24": 0.1,
    "V25": -0.5, "V26": 0.4, "V27": -0.2, "V28": 0.1,
}

NORMAL_TXN = {
    "Time": 36000, "Amount": 45.50,
    **{f"V{i}": 0.1 for i in range(1, 29)},
}

# ── FeatureAgent Tests ─────────────────────────────────────────────────────────
class TestFeatureAgent:
    def setup_method(self):
        self.agent = FeatureAgent()

    def test_extracts_33_features(self):
        result = self.agent.run(SAMPLE_TXN)
        assert result.success
        assert len(result.output["features"]) == 33

    def test_detects_night_time(self):
        result = self.agent.run(SAMPLE_TXN)
        assert result.output["features"]["Is_night"] == 1

    def test_high_amount_flag(self):
        result = self.agent.run(SAMPLE_TXN)
        assert result.output["features"]["High_amount"] == 1

    def test_daytime_low_amount_no_flags(self):
        result = self.agent.run(NORMAL_TXN)
        assert result.output["features"]["Is_night"] == 0
        assert result.output["features"]["High_amount"] == 0

    def test_missing_field_raises_error(self):
        bad_txn = {"Time": 100, "Amount": 50}  # missing V1..V28
        result = self.agent.run(bad_txn)
        assert not result.success

# ── ScorerAgent Tests ──────────────────────────────────────────────────────────
class TestScorerAgent:
    def setup_method(self):
        self.feature_agent = FeatureAgent()
        self.scorer        = ScorerAgent()

    def _get_features(self, txn):
        return self.feature_agent.run(txn).output

    def test_fraud_score_high_for_suspicious(self):
        feat = self._get_features(SAMPLE_TXN)
        result = self.scorer.run(feat)
        assert result.success
        assert result.output["fraud_probability"] > 0.5

    def test_fraud_score_low_for_normal(self):
        feat = self._get_features(NORMAL_TXN)
        result = self.scorer.run(feat)
        assert result.success
        assert result.output["fraud_probability"] < 0.5

    def test_risk_tier_critical_for_high_fraud(self):
        feat = self._get_features(SAMPLE_TXN)
        result = self.scorer.run(feat)
        assert result.output["risk_tier"] in ["HIGH", "CRITICAL"]

    def test_top_features_returned(self):
        feat = self._get_features(NORMAL_TXN)
        result = self.scorer.run(feat)
        assert len(result.output["top_features"]) == 5

# ── ExplanationAgent Tests ─────────────────────────────────────────────────────
class TestExplanationAgent:
    def setup_method(self):
        self.agent = ExplanationAgent()

    def test_fallback_explanation_generated(self):
        score_data = {
            "fraud_probability": 0.95,
            "risk_tier": "CRITICAL",
            "risk_signals": ["High amount ($1249.99)", "Night-time transaction"],
            "top_features": [],
            "raw": {"Amount": 1249.99},
        }
        result = self.agent.run(score_data)
        assert result.success
        assert len(result.output["explanation"]) > 10

    def test_low_risk_explanation_contains_approve(self):
        score_data = {
            "fraud_probability": 0.02,
            "risk_tier": "LOW",
            "risk_signals": [],
            "top_features": [],
            "raw": {"Amount": 45.0},
        }
        result = self.agent.run(score_data)
        assert "APPROVE" in result.output["explanation"].upper() or \
               "legitimate" in result.output["explanation"].lower()

# ── AggregatorAgent Tests ──────────────────────────────────────────────────────
class TestAggregatorAgent:
    def setup_method(self):
        self.agent = AggregatorAgent()

    def test_block_verdict_for_critical(self):
        inputs = {
            "score_data": {
                "fraud_probability": 0.98,
                "risk_tier": "CRITICAL",
                "risk_signals": ["High amount"],
                "top_features": [],
                "raw": {},
            },
            "explanation_data": {
                "explanation": "Fraud detected.",
                "risk_tier": "CRITICAL",
                "fraud_probability": 0.98,
                "risk_signals": [],
            },
            "transaction_id": "TEST_001",
        }
        result = self.agent.run(inputs)
        assert result.success
        assert result.output["final_verdict"] == "BLOCK"

    def test_approve_verdict_for_low(self):
        inputs = {
            "score_data": {
                "fraud_probability": 0.05,
                "risk_tier": "LOW",
                "risk_signals": [],
                "top_features": [],
                "raw": {},
            },
            "explanation_data": {
                "explanation": "Legitimate.",
                "risk_tier": "LOW",
                "fraud_probability": 0.05,
                "risk_signals": [],
            },
            "transaction_id": "TEST_002",
        }
        result = self.agent.run(inputs)
        assert result.output["final_verdict"] == "APPROVE"

# ── MCP Server Tests ───────────────────────────────────────────────────────────
class TestMCPServer:
    def setup_method(self):
        self.mcp = MCPServer()

    def test_get_stats_returns_dict(self):
        stats = self.mcp.call("get_stats")
        assert isinstance(stats, dict)

    def test_flag_for_review(self):
        result = self.mcp.call("flag_for_review", transaction_id="TEST_999", reason="Test flag")
        assert result["status"] == "flagged"
        assert result["transaction_id"] == "TEST_999"

    def test_unknown_tool_raises(self):
        with pytest.raises(ValueError):
            self.mcp.call("nonexistent_tool")

    def test_query_audit_returns_list(self):
        records = self.mcp.call("query_audit_log", limit=5)
        assert isinstance(records, list)

# ── End-to-end Orchestrator Test ───────────────────────────────────────────────
class TestOrchestrator:
    def setup_method(self):
        self.orch = OrchestratorAgent()

    def test_full_pipeline_returns_verdict(self):
        result = self.orch.analyze(SAMPLE_TXN, "E2E_TEST_001")
        assert "verdict" in result
        assert result["verdict"] in ["APPROVE","REVIEW","HOLD","BLOCK","ERROR"]

    def test_suspicious_txn_flagged(self):
        result = self.orch.analyze(SAMPLE_TXN)
        assert result["verdict"] in ["HOLD","BLOCK"]

    def test_normal_txn_approved(self):
        result = self.orch.analyze(NORMAL_TXN)
        assert result["verdict"] in ["APPROVE","REVIEW"]
