"""
Alert Agent — final stage of the pipeline. Takes the aggregated verdict
and generates structured alert reports, logs to audit file, and
produces console/dashboard-ready output.
"""
import json
import os
from datetime import datetime
from agents.base_agent import BaseAgent

AUDIT_LOG_PATH = "data/audit_log.jsonl"


class AlertAgent(BaseAgent):
    """
    Responsibilities:
    - Format the final verdict into a human-readable alert
    - Persist every decision to an append-only audit log (JSONL)
    - Return structured alert object for the dashboard
    """

    SEVERITY_EMOJI = {
        "APPROVE": "✅",
        "REVIEW":  "🟡",
        "HOLD":    "🟠",
        "BLOCK":   "🔴",
    }

    def __init__(self):
        super().__init__("AlertAgent")
        os.makedirs("data", exist_ok=True)

    def _execute(self, record: dict) -> dict:
        verdict  = record["final_verdict"]
        txn_id   = record["transaction_id"]
        prob     = record["fraud_probability"]
        tier     = record["risk_tier"]
        emoji    = self.SEVERITY_EMOJI.get(verdict, "⚪")

        # Build formatted alert
        alert_text = self._format_alert(record, emoji)

        # Persist to audit log
        self._write_audit(record)

        self.log(f"{emoji} Alert issued: {verdict} | {txn_id}")
        print("\n" + "=" * 60)
        print(alert_text)
        print("=" * 60 + "\n")

        return {
            "alert_text":   alert_text,
            "verdict":      verdict,
            "tier":         tier,
            "probability":  prob,
            "transaction_id": txn_id,
            "timestamp":    record["timestamp"],
            "explanation":  record["explanation"],
            "risk_signals": record["risk_signals"],
            "top_features": record["top_features"],
            "emoji":        emoji,
        }

    def _format_alert(self, record: dict, emoji: str) -> str:
        raw = record.get("raw_transaction", {})
        amount = raw.get("Amount", "N/A")
        lines = [
            f"{emoji} FRAUD DETECTION ALERT",
            f"Transaction ID : {record['transaction_id']}",
            f"Timestamp      : {record['timestamp']}",
            f"Amount         : ${amount}",
            f"Fraud Prob.    : {record['fraud_probability']:.1%}",
            f"Risk Tier      : {record['risk_tier']}",
            f"Final Verdict  : {record['final_verdict']}",
            f"Confidence     : {record['confidence']:.1%}",
            "",
            "Explanation:",
            record['explanation'],
            "",
            "Risk Signals:",
        ] + [f"  • {s}" for s in record.get("risk_signals", [])] + [
            "",
            "Top Features:",
        ] + [f"  • {f['feature']} (importance={f['importance']})"
             for f in record.get("top_features", [])]
        return "\n".join(lines)

    def _write_audit(self, record: dict):
        """Appends record to JSONL audit log for reproducibility."""
        try:
            with open(AUDIT_LOG_PATH, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            self.log(f"Warning: could not write audit log: {e}")
