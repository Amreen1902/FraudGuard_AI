"""
Orchestrator Agent — the central coordinator of the fraud detection
multi-agent system. Routes transactions through the full pipeline:

  Transaction → FeatureAgent → ScorerAgent → ExplanationAgent
             → AggregatorAgent → AlertAgent → Result

Also exposes batch processing and uses the MCP Server for audit queries.
"""
import uuid
import pandas as pd
from agents.base_agent import BaseAgent, AgentResult
from agents.feature_agent import FeatureAgent
from agents.scorer_agent import ScorerAgent
from agents.explanation_agent import ExplanationAgent
from agents.aggregator_agent import AggregatorAgent
from agents.alert_agent import AlertAgent
from tools.mcp_server import MCPServer


class OrchestratorAgent(BaseAgent):
    """
    Manages the full fraud detection pipeline.
    Supports:
      - Single transaction analysis: analyze(transaction_dict)
      - Batch CSV analysis:          analyze_batch(csv_path, n_samples)
      - MCP tool queries:            query_mcp(tool_name, **kwargs)
    """

    def __init__(self):
        super().__init__("OrchestratorAgent")

        # Instantiate all sub-agents
        self.feature_agent     = FeatureAgent()
        self.scorer_agent      = ScorerAgent()
        self.explanation_agent = ExplanationAgent()
        self.aggregator_agent  = AggregatorAgent()
        self.alert_agent       = AlertAgent()

        # MCP server for audit/database tools
        self.mcp = MCPServer()

        self.log("All sub-agents initialized ✓")

    # ------------------------------------------------------------------ #
    #  Single transaction                                                  #
    # ------------------------------------------------------------------ #
    def analyze(self, transaction: dict, transaction_id: str = None) -> dict:
        """
        Runs a single transaction through the full pipeline.

        Args:
            transaction:    dict with Time, Amount, V1..V28
            transaction_id: optional custom ID

        Returns:
            Final alert dict from AlertAgent
        """
        txn_id = transaction_id or f"TXN_{uuid.uuid4().hex[:10].upper()}"
        self.log(f"▶ Pipeline start | {txn_id}")

        # Step 1 — Feature Engineering
        feat_result = self.feature_agent.run(transaction)
        if not feat_result.success:
            return self._error_response(txn_id, "FeatureAgent", feat_result.error)

        # Step 2 — ML Scoring
        score_result = self.scorer_agent.run(feat_result.output)
        if not score_result.success:
            return self._error_response(txn_id, "ScorerAgent", score_result.error)

        # Step 3 — LLM Explanation
        expl_result = self.explanation_agent.run(score_result.output)
        if not expl_result.success:
            return self._error_response(txn_id, "ExplanationAgent", expl_result.error)

        # Step 4 — Aggregation
        agg_result = self.aggregator_agent.run({
            "score_data":       score_result.output,
            "explanation_data": expl_result.output,
            "transaction_id":   txn_id,
        })
        if not agg_result.success:
            return self._error_response(txn_id, "AggregatorAgent", agg_result.error)

        # Step 5 — Alert
        alert_result = self.alert_agent.run(agg_result.output)
        if not alert_result.success:
            return self._error_response(txn_id, "AlertAgent", alert_result.error)

        self.log(f"✓ Pipeline complete | {txn_id} → {alert_result.output['verdict']}")
        return alert_result.output

    # ------------------------------------------------------------------ #
    #  Batch processing                                                    #
    # ------------------------------------------------------------------ #
    def analyze_batch(self, csv_path: str, n_samples: int = 20) -> list[dict]:
        """
        Reads a CSV file and runs the pipeline on n_samples transactions.
        Prioritizes flagged/fraud-likely rows for a useful demo.

        Args:
            csv_path:  path to transactions CSV
            n_samples: number of transactions to process

        Returns:
            List of alert dicts
        """
        self.log(f"Batch mode | {csv_path} | n={n_samples}")
        df = pd.read_csv(csv_path)

        # Sample: half fraud-likely (V1 < -2), half random
        fraud_like = df[df["V1"] < -2].sample(
            min(n_samples // 2, len(df[df["V1"] < -2])), random_state=42
        )
        legit_like = df[df["V1"] >= -2].sample(
            min(n_samples - len(fraud_like), len(df[df["V1"] >= -2])), random_state=42
        )
        sample = pd.concat([fraud_like, legit_like]).sample(frac=1, random_state=42)

        results = []
        for _, row in sample.iterrows():
            txn = row.to_dict()
            txn.pop("Class", None)  # remove label if present
            result = self.analyze(txn)
            results.append(result)

        self.log(f"Batch complete | {len(results)} transactions processed")
        return results

    # ------------------------------------------------------------------ #
    #  MCP Tool access                                                     #
    # ------------------------------------------------------------------ #
    def query_mcp(self, tool_name: str, **kwargs):
        """
        Exposes MCP server tools to external callers and other agents.
        e.g. orchestrator.query_mcp("get_stats")
        """
        self.log(f"MCP call: {tool_name} | args={kwargs}")
        return self.mcp.call(tool_name, **kwargs)

    # ------------------------------------------------------------------ #
    #  Required BaseAgent method (for run() compatibility)                 #
    # ------------------------------------------------------------------ #
    def _execute(self, input_data):
        """Delegates to analyze() when called via run() interface."""
        if isinstance(input_data, dict):
            return self.analyze(input_data)
        raise ValueError("OrchestratorAgent._execute expects a transaction dict")

    def _error_response(self, txn_id: str, agent: str, error: str) -> dict:
        return {
            "transaction_id": txn_id,
            "verdict":        "ERROR",
            "tier":           "UNKNOWN",
            "probability":    None,
            "error":          f"{agent} failed: {error}",
            "explanation":    "Pipeline error — manual review required.",
            "emoji":          "⚠️",
        }
