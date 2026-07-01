"""
ML Scorer Agent — loads the trained XGBoost model and predicts fraud probability.
Returns a structured score with confidence tier and top contributing features.
"""
import pickle
import numpy as np
import pandas as pd
from agents.base_agent import BaseAgent

MODEL_PATH = "models/fraud_model.pkl"
SCALER_PATH = "models/scaler.pkl"
FEATURES_PATH = "models/feature_names.pkl"


class ScorerAgent(BaseAgent):
    """
    Loads the persisted XGBoost + scaler, runs inference, and returns:
    - fraud_probability (0.0 – 1.0)
    - risk_tier: LOW / MEDIUM / HIGH / CRITICAL
    - top_features: top 5 features driving the prediction
    """

    THRESHOLDS = {
        "LOW": 0.25,
        "MEDIUM": 0.50,
        "HIGH": 0.75,
        "CRITICAL": 1.01,
    }

    def __init__(self):
        super().__init__("ScorerAgent")
        self._model = None
        self._scaler = None
        self._feature_names = None
        self._load_artifacts()

    def _load_artifacts(self):
        with open(MODEL_PATH, "rb") as f:
            self._model = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            self._scaler = pickle.load(f)
        with open(FEATURES_PATH, "rb") as f:
            self._feature_names = pickle.load(f)
        self.log(f"Loaded model | {len(self._feature_names)} features")

    def _execute(self, feature_data: dict) -> dict:
        features = feature_data["features"]

        # Build input row in correct feature order
        row = np.array([[features[f] for f in self._feature_names]])
        row_scaled = self._scaler.transform(row)

        prob = float(self._model.predict_proba(row_scaled)[0][1])
        prediction = int(prob >= 0.5)
        tier = self._get_tier(prob)

        # Feature importance (SHAP-style approximation using model importances)
        importances = self._model.feature_importances_
        feat_imp = sorted(
            zip(self._feature_names, importances),
            key=lambda x: x[1],
            reverse=True,
        )[:5]

        self.log(f"Fraud probability: {prob:.4f} | Tier: {tier}")

        return {
            "fraud_probability": round(prob, 4),
            "prediction": prediction,
            "risk_tier": tier,
            "top_features": [{"feature": f, "importance": round(float(v), 4)}
                             for f, v in feat_imp],
            "features": features,
            "risk_signals": feature_data.get("risk_signals", []),
            "raw": feature_data.get("raw", {}),
        }

    def _get_tier(self, prob: float) -> str:
        if prob < 0.25:
            return "LOW"
        elif prob < 0.50:
            return "MEDIUM"
        elif prob < 0.75:
            return "HIGH"
        else:
            return "CRITICAL"
