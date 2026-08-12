"""
Wires the generator and detector together on a background thread and holds
bounded, thread-safe buffers of recent trades/alerts for the Flask app to
read from (both for the SSE live stream and the plain REST endpoints used
on initial dashboard load).
"""
from __future__ import annotations

import csv
import json
import os
import queue
import threading
import time
from collections import deque
from typing import Optional

from .detector import AnomalyPipelineDetector
from .generator import TradeGenerator


class AnomalyPipeline:
    def __init__(
        self,
        ticks_per_second: float = 8.0,
        anomaly_probability: float = 0.015,
        history_size: int = 500,
        log_path: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        self.generator = TradeGenerator(
            ticks_per_second=ticks_per_second,
            anomaly_probability=anomaly_probability,
            seed=seed,
        )
        self.detector = AnomalyPipelineDetector()

        self.trade_history = deque(maxlen=history_size)
        self.alert_history = deque(maxlen=history_size)
        self._lock = threading.Lock()

        self.stats = {
            "trades_seen": 0,
            "alerts_raised": 0,
            "true_positive_injected": 0,   # injected anomaly AND flagged
            "false_negative_injected": 0,  # injected anomaly, NOT flagged
            "false_positive_flagged": 0,   # flagged but NOT injected
        }

        # Fan-out: each connected SSE client gets its own Queue subscribed here.
        self._subscribers: list[queue.Queue] = []

        self._log_path = log_path
        if self._log_path:
            os.makedirs(os.path.dirname(self._log_path) or ".", exist_ok=True)
            if not os.path.exists(self._log_path):
                with open(self._log_path, "w", newline="") as f:
                    csv.writer(f).writerow(
                        ["timestamp", "symbol", "price", "volume", "flagged", "reasons"]
                    )

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    # ---- pub/sub for SSE ----

    def subscribe(self) -> "queue.Queue":
        q: queue.Queue = queue.Queue(maxsize=1000)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: "queue.Queue"):
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def _publish(self, event: dict):
        with self._lock:
            subs = list(self._subscribers)
        for q in subs:
            try:
                q.put_nowait(event)
            except queue.Full:
                pass  # slow client — drop rather than block the pipeline

    # ---- main loop ----

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _run(self):
        for trade in self.generator.stream():
            if self._stop.is_set():
                break

            alert = self.detector.evaluate(trade)

            with self._lock:
                self.stats["trades_seen"] += 1
                self.trade_history.append(trade.to_dict())
                if alert:
                    self.stats["alerts_raised"] += 1
                    if trade.is_injected_anomaly:
                        self.stats["true_positive_injected"] += 1
                    else:
                        self.stats["false_positive_flagged"] += 1
                elif trade.is_injected_anomaly:
                    self.stats["false_negative_injected"] += 1

                if alert:
                    self.alert_history.append(alert.to_dict())

            if self._log_path:
                with open(self._log_path, "a", newline="") as f:
                    csv.writer(f).writerow([
                        trade.timestamp, trade.symbol, trade.price, trade.volume,
                        bool(alert), "; ".join(alert.reasons) if alert else "",
                    ])

            self._publish({
                "type": "alert" if alert else "trade",
                "trade": trade.to_dict(),
                "alert": alert.to_dict() if alert else None,
            })

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "stats": dict(self.stats),
                "recent_trades": list(self.trade_history)[-100:],
                "recent_alerts": list(self.alert_history)[-100:],
            }
