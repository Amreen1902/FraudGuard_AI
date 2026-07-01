"""
Decision Aggregator Agent — combines ML score + LLM explanation into a
final structured verdict with recommended action and audit trail.
"""
from datetime import datetime
from agents.base_agent import BaseAgent


class AggregatorAgent(BaseAgent):
    """
    Takes outputs from ScorerAgent + ExplanationAgent and produces:
    - final_verdict: APPROVE / REVIEW / HOLD / BLOCK
    - confidence_score
    - full audit record for logging/dashboard
    """

    ACTION_MAP = {
        "LOW":      "APPROVE",
        "MEDIUM":   "REVIEW",
        "HIGH":     "HOLD",
        "CRITICAL": "BLOCK",
    }

    def __init__(self):
        super().__init__("AggregatorAgent")

    def _execute(self, inputs: dict) -> dict:
        """
        inputs = {
            "score_data":       output of ScorerAgent._execute()
            "explanation_data": output of ExplanationAgent._execute()
            "transaction_id":   optional string
        }
        """
        score = inputs["score_data"]
        expl  = inputs["explanation_data"]
        txn_id = inputs.get("transaction_id", f"TXN_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}")

        tier    = score["risk_tier"]
        prob    = score["fraud_probability"]
        verdict = self.ACTION_MAP[tier]

        # Confidence = how far the probability is from the 0.5 boundary
        confidence = round(abs(prob - 0.5) * 2, 4)  # 0.0 – 1.0

        self.log(f"Transaction {txn_id} | verdict={verdict} | confidence={confidence:.2%}")

        record = {
            "transaction_id":    txn_id,
            "timestamp":         datetime.utcnow().isoformat() + "Z",
            "fraud_probability": prob,
            "risk_tier":         tier,
            "final_verdict":     verdict,
            "confidence":        confidence,
            "explanation":       expl["explanation"],
            "risk_signals":      score.get("risk_signals", []),
            "top_features":      score.get("top_features", []),
            "raw_transaction":   score.get("raw", {}),
        }
        return record
