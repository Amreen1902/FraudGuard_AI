"""
Feature Agent — takes raw transaction dict/row and engineers features
that the ML model expects. This is the first step in the pipeline.
"""
import numpy as np
import pandas as pd
from agents.base_agent import BaseAgent


class FeatureAgent(BaseAgent):
    """
    Responsible for:
    - Validating incoming transaction fields
    - Engineering derived features (log amount, hour-of-day, night flag, etc.)
    - Returning a feature dict ready for the ML Scorer Agent
    """

    REQUIRED_FIELDS = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]

    def __init__(self):
        super().__init__("FeatureAgent")

    def _execute(self, transaction: dict) -> dict:
        # --- Validation ---
        missing = [f for f in self.REQUIRED_FIELDS if f not in transaction]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")

        self.log(f"Processing transaction | Amount=${transaction['Amount']:.2f}")

        # --- Feature engineering ---
        features = {}

        # Copy V1..V28 directly
        for i in range(1, 29):
            features[f"V{i}"] = float(transaction[f"V{i}"])

        # Derived features (must match models/train.py)
        features["Amount_log"] = np.log1p(float(transaction["Amount"]))
        hour = (float(transaction["Time"]) % 86400) // 3600
        features["Hour"] = hour
        features["Is_night"] = int(hour >= 22 or hour <= 5)
        features["V1_V3_ratio"] = features["V1"] / (abs(features["V3"]) + 1e-5)
        features["High_amount"] = int(float(transaction["Amount"]) > 500)

        # Risk signals for explanation agent
        risk_signals = self._extract_risk_signals(transaction, features)

        self.log(f"Engineered {len(features)} features | "
                 f"Risk signals: {risk_signals}")

        return {
            "features": features,
            "risk_signals": risk_signals,
            "raw": transaction,
        }

    def _extract_risk_signals(self, txn: dict, features: dict) -> list[str]:
        """Identifies human-readable risk indicators for the Explanation Agent."""
        signals = []
        if txn["Amount"] > 500:
            signals.append(f"High amount (${txn['Amount']:.2f})")
        if features["Is_night"]:
            signals.append(f"Night-time transaction (hour {int(features['Hour'])}:00)")
        if features["V1"] < -2.5:
            signals.append("Unusual V1 pattern (strong negative deviation)")
        if features["V3"] > 2.5:
            signals.append("Unusual V3 pattern (strong positive deviation)")
        if abs(features["V9"]) > 2.0:
            signals.append("Unusual V9 pattern")
        if not signals:
            signals.append("No major individual risk signals detected")
        return signals
