"""
Offline evaluation: runs N simulated trades through the detector (no Flask,
no real-time pacing) and reports precision/recall against the ground-truth
injected anomalies. Useful for tuning thresholds and for demonstrating that
the pipeline actually catches what it's supposed to.

Usage:
    python3 eval_detector.py --trades 5000 --anomaly-prob 0.02
"""
from __future__ import annotations

import argparse
import itertools
import time

from app.detector import AnomalyPipelineDetector
from app.generator import TradeGenerator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trades", type=int, default=5000)
    parser.add_argument("--anomaly-prob", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    gen = TradeGenerator(anomaly_probability=args.anomaly_prob, seed=args.seed)
    detector = AnomalyPipelineDetector()

    tp = fp = fn = tn = 0
    by_kind = {}
    t0 = time.time()

    for trade in itertools.islice(gen.stream(paced=False), args.trades):
        alert = detector.evaluate(trade)
        flagged = alert is not None
        injected = trade.is_injected_anomaly

        if injected and flagged:
            tp += 1
            by_kind.setdefault(trade.anomaly_kind, [0, 0])[0] += 1
        elif injected and not flagged:
            fn += 1
            by_kind.setdefault(trade.anomaly_kind, [0, 0])[1] += 1
        elif not injected and flagged:
            fp += 1
        else:
            tn += 1

    elapsed = time.time() - t0
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else float("nan")

    print(f"processed {args.trades} trades in {elapsed:.2f}s ({args.trades/elapsed:.0f} trades/sec)")
    print()
    print(f"  true positives (caught injected anomalies): {tp}")
    print(f"  false negatives (missed injected anomalies): {fn}")
    print(f"  false positives (flagged normal trades):      {fp}")
    print(f"  true negatives:                                {tn}")
    print()
    print(f"  precision: {precision:.3f}   recall: {recall:.3f}   f1: {f1:.3f}")
    print()
    print("  recall by injected anomaly type (caught / missed):")
    for kind, (caught, missed) in sorted(by_kind.items()):
        total = caught + missed
        print(f"    {kind:15s} {caught}/{total} ({caught/total:.1%})")


if __name__ == "__main__":
    main()
