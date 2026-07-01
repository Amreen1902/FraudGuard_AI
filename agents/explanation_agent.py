"""
Explanation Agent — uses Gemini API to generate a clear,
human-readable explanation of why a transaction was flagged as fraud.
This is the LLM-powered reasoning layer of the multi-agent system.
"""
import os
import requests
from agents.base_agent import BaseAgent

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"


class ExplanationAgent(BaseAgent):
    """
    Calls Gemini API with ML score + risk signals → returns plain-English explanation.
    Falls back to a rule-based explanation if API key is unavailable.
    """

    def __init__(self):
        super().__init__("ExplanationAgent")
        self._api_key = os.getenv("GEMINI_API_KEY", "")

    def _execute(self, score_data: dict) -> dict:
        prob = score_data["fraud_probability"]
        tier = score_data["risk_tier"]
        signals = score_data.get("risk_signals", [])
        top_feats = score_data.get("top_features", [])
        raw = score_data.get("raw", {})

        self.log(f"Generating explanation | tier={tier} prob={prob:.2%}")

        if self._api_key:
            explanation = self._call_llm(prob, tier, signals, top_feats, raw)
        else:
            self.log("No API key found — using rule-based fallback explanation")
            explanation = self._rule_based_explanation(prob, tier, signals)

        return {
            "explanation": explanation,
            "risk_tier": tier,
            "fraud_probability": prob,
            "risk_signals": signals,
        }

    def _call_llm(self, prob, tier, signals, top_feats, raw) -> str:
        """Calls Gemini API to generate a detailed fraud explanation."""
        amount = raw.get("Amount", "unknown")
        feat_str = ", ".join(
            f"{f['feature']} (importance={f['importance']})" for f in top_feats
        )
        signal_str = "\n".join(f"  - {s}" for s in signals)

        prompt = f"""You are a fraud analyst AI assistant. Analyze this transaction and explain the decision clearly.

Transaction Details:
- Amount: ${amount}
- Fraud Probability: {prob:.1%}
- Risk Tier: {tier}

Risk Signals Detected:
{signal_str}

Top Contributing ML Features:
  {feat_str}

Write a concise, professional fraud analysis report (3-4 sentences) that:
1. States the risk verdict clearly
2. Explains the main reasons for suspicion in plain language
3. Recommends an action (approve / review / block)

Keep it under 100 words. Do not use bullet points."""

        resp = requests.post(
            f"{GEMINI_API_URL}?key={self._api_key}",
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()

    def _rule_based_explanation(self, prob: float, tier: str, signals: list) -> str:
        """Fallback explanation when API key is not available."""
        verdict_map = {
            "LOW":      "This transaction appears legitimate with low fraud risk.",
            "MEDIUM":   "This transaction shows some suspicious patterns and warrants review.",
            "HIGH":     "This transaction has a high probability of fraud and should be flagged.",
            "CRITICAL": "This transaction is almost certainly fraudulent and must be blocked immediately.",
        }
        verdict = verdict_map.get(tier, "Risk level unknown.")
        signal_text = "; ".join(signals[:2]) if signals else "No specific signals"
        action_map = {
            "LOW":      "Recommended action: APPROVE.",
            "MEDIUM":   "Recommended action: MANUAL REVIEW.",
            "HIGH":     "Recommended action: HOLD for investigation.",
            "CRITICAL": "Recommended action: BLOCK immediately.",
        }
        return (
            f"{verdict} Fraud probability: {prob:.1%}. "
            f"Key signals: {signal_text}. "
            f"{action_map.get(tier, '')}"
        )