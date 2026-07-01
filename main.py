"""
main.py — Entry point for the Fraud Detection Multi-Agent System
Usage:
  python main.py                        # demo mode (5 sample transactions)
  python main.py --batch 20             # batch mode (20 transactions from CSV)
  python main.py --stats                # show audit log stats via MCP
  python main.py --single               # analyze one hardcoded sample
"""
import argparse
import json
from agents.orchestrator_agent import OrchestratorAgent

DEMO_TRANSACTIONS = [
    # Suspicious transaction — high amount, night time, unusual V1
    {
        "transaction_id": "TXN_DEMO_001",
        "transaction": {
            "Time": 82800, "Amount": 1249.99,
            "V1": -3.8, "V2": 1.9, "V3": -1.1, "V4": 2.3,
            "V5": -0.5, "V6": -0.8, "V7": 0.4, "V8": 0.1,
            "V9": -2.2, "V10": -0.3, "V11": 1.1, "V12": -1.9,
            "V13": 0.2, "V14": -2.1, "V15": 0.5, "V16": -0.7,
            "V17": -0.9, "V18": 0.3, "V19": -0.1, "V20": 0.6,
            "V21": -0.4, "V22": 0.2, "V23": -0.3, "V24": 0.1,
            "V25": -0.5, "V26": 0.4, "V27": -0.2, "V28": 0.1,
        }
    },
    # Normal transaction — low amount, daytime
    {
        "transaction_id": "TXN_DEMO_002",
        "transaction": {
            "Time": 36000, "Amount": 45.50,
            "V1": 0.2, "V2": -0.3, "V3": 0.4, "V4": -0.1,
            "V5": 0.3, "V6": 0.1, "V7": -0.2, "V8": 0.0,
            "V9": 0.1, "V10": 0.2, "V11": -0.1, "V12": 0.3,
            "V13": -0.2, "V14": 0.1, "V15": 0.0, "V16": -0.1,
            "V17": 0.2, "V18": -0.3, "V19": 0.1, "V20": 0.0,
            "V21": -0.1, "V22": 0.2, "V23": 0.0, "V24": -0.1,
            "V25": 0.1, "V26": 0.0, "V27": -0.2, "V28": 0.0,
        }
    },
    # Medium risk — moderate amount, slightly unusual features
    {
        "transaction_id": "TXN_DEMO_003",
        "transaction": {
            "Time": 54000, "Amount": 380.00,
            "V1": -1.5, "V2": 0.8, "V3": -0.5, "V4": 1.2,
            "V5": -0.2, "V6": -0.3, "V7": 0.1, "V8": 0.2,
            "V9": -0.9, "V10": -0.1, "V11": 0.5, "V12": -0.8,
            "V13": 0.1, "V14": -0.9, "V15": 0.2, "V16": -0.3,
            "V17": -0.4, "V18": 0.1, "V19": 0.0, "V20": 0.3,
            "V21": -0.2, "V22": 0.1, "V23": -0.1, "V24": 0.0,
            "V25": -0.2, "V26": 0.1, "V27": -0.1, "V28": 0.0,
        }
    },
]


def run_demo(orchestrator: OrchestratorAgent):
    print("\n🤖 FRAUD DETECTION MULTI-AGENT SYSTEM — DEMO MODE")
    print("=" * 60)
    results = []
    for item in DEMO_TRANSACTIONS:
        result = orchestrator.analyze(item["transaction"], item["transaction_id"])
        results.append(result)
    return results


def run_batch(orchestrator: OrchestratorAgent, n: int):
    print(f"\n🤖 BATCH MODE — processing {n} transactions from CSV")
    print("=" * 60)
    return orchestrator.analyze_batch("data/transactions.csv", n_samples=n)


def run_stats(orchestrator: OrchestratorAgent):
    print("\n📊 MCP SERVER — AUDIT STATISTICS")
    print("=" * 60)
    stats = orchestrator.query_mcp("get_stats")
    print(json.dumps(stats, indent=2))
    recent = orchestrator.query_mcp("query_audit_log", verdict="BLOCK", limit=5)
    print(f"\nMost recent BLOCK decisions ({len(recent)}):")
    for r in recent:
        print(f"  • {r['transaction_id']} | prob={r['fraud_probability']:.1%} | {r['timestamp']}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Fraud Detection Agent System")
    parser.add_argument("--batch",  type=int, metavar="N", help="Run batch mode with N transactions")
    parser.add_argument("--stats",  action="store_true",    help="Show MCP audit stats")
    parser.add_argument("--single", action="store_true",    help="Analyze one sample transaction")
    args = parser.parse_args()

    orchestrator = OrchestratorAgent()

    if args.stats:
        run_stats(orchestrator)
    elif args.batch:
        results = run_batch(orchestrator, args.batch)
        verdicts = [r.get("verdict", "ERROR") for r in results]
        print(f"\nSummary: {dict(zip(*([v for v in set(verdicts)], [verdicts.count(v) for v in set(verdicts)])))} ")
    elif args.single:
        orchestrator.analyze(DEMO_TRANSACTIONS[0]["transaction"], "TXN_SINGLE_TEST")
    else:
        run_demo(orchestrator)

    print("\n✅ Done. Run 'streamlit run dashboard/app.py' to view the dashboard.")


if __name__ == "__main__":
    main()
