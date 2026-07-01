"""
MCP (Model Context Protocol) Server
Provides tool-based access to the fraud audit log and transaction database.
Agents call these tools to query historical fraud decisions.

Implements the MCP concept as a lightweight JSON-RPC style server
that can be run standalone or imported by the Orchestrator.
"""
import json
import os
from datetime import datetime

AUDIT_LOG_PATH = "data/audit_log.jsonl"


class MCPServer:
    """
    MCP Tool Server — exposes the following tools to agents:
      1. query_audit_log   — search past decisions by verdict/tier
      2. get_transaction   — look up a specific transaction by ID
      3. get_stats         — aggregate fraud stats from audit log
      4. flag_for_review   — manually flag a transaction
    """

    def __init__(self):
        self.tools = {
            "query_audit_log":  self.query_audit_log,
            "get_transaction":  self.get_transaction,
            "get_stats":        self.get_stats,
            "flag_for_review":  self.flag_for_review,
        }
        self._flagged: list[dict] = []
        print("[MCPServer] Initialized with tools:", list(self.tools.keys()))

    # ------------------------------------------------------------------ #
    #  Tool: query_audit_log                                               #
    # ------------------------------------------------------------------ #
    def query_audit_log(
        self,
        verdict: str = None,
        tier: str = None,
        limit: int = 20,
    ) -> list[dict]:
        """
        Returns recent audit records filtered by verdict and/or risk tier.

        Args:
            verdict: APPROVE | REVIEW | HOLD | BLOCK  (optional)
            tier:    LOW | MEDIUM | HIGH | CRITICAL    (optional)
            limit:   max records to return
        """
        records = self._load_audit()
        if verdict:
            records = [r for r in records if r.get("final_verdict") == verdict.upper()]
        if tier:
            records = [r for r in records if r.get("risk_tier") == tier.upper()]
        return records[-limit:]

    # ------------------------------------------------------------------ #
    #  Tool: get_transaction                                               #
    # ------------------------------------------------------------------ #
    def get_transaction(self, transaction_id: str) -> dict | None:
        """
        Retrieves a specific transaction record by its ID.

        Args:
            transaction_id: e.g. "TXN_20240601_001"
        """
        records = self._load_audit()
        for r in records:
            if r.get("transaction_id") == transaction_id:
                return r
        return None

    # ------------------------------------------------------------------ #
    #  Tool: get_stats                                                     #
    # ------------------------------------------------------------------ #
    def get_stats(self) -> dict:
        """
        Returns aggregate fraud detection statistics from the audit log.
        """
        records = self._load_audit()
        if not records:
            return {"total": 0, "message": "No records in audit log yet."}

        total = len(records)
        by_verdict: dict[str, int] = {}
        by_tier:    dict[str, int] = {}
        probs = []

        for r in records:
            v = r.get("final_verdict", "UNKNOWN")
            t = r.get("risk_tier",     "UNKNOWN")
            by_verdict[v] = by_verdict.get(v, 0) + 1
            by_tier[t]    = by_tier.get(t, 0)    + 1
            probs.append(r.get("fraud_probability", 0))

        return {
            "total_transactions": total,
            "by_verdict":         by_verdict,
            "by_tier":            by_tier,
            "avg_fraud_probability": round(sum(probs) / len(probs), 4),
            "fraud_rate":         round(
                (by_verdict.get("BLOCK", 0) + by_verdict.get("HOLD", 0)) / total, 4
            ),
        }

    # ------------------------------------------------------------------ #
    #  Tool: flag_for_review                                               #
    # ------------------------------------------------------------------ #
    def flag_for_review(self, transaction_id: str, reason: str = "") -> dict:
        """
        Manually flags a transaction for human review.

        Args:
            transaction_id: ID to flag
            reason:         optional reason string
        """
        entry = {
            "transaction_id": transaction_id,
            "flagged_at":     datetime.utcnow().isoformat() + "Z",
            "reason":         reason,
        }
        self._flagged.append(entry)
        return {"status": "flagged", **entry}

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #
    def call(self, tool_name: str, **kwargs):
        """Generic dispatcher — agents call this to invoke any MCP tool."""
        if tool_name not in self.tools:
            raise ValueError(f"Unknown MCP tool: '{tool_name}'. "
                             f"Available: {list(self.tools.keys())}")
        return self.tools[tool_name](**kwargs)

    def _load_audit(self) -> list[dict]:
        if not os.path.exists(AUDIT_LOG_PATH):
            return []
        records = []
        with open(AUDIT_LOG_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return records
