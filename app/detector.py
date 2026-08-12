"""
Anomaly detection over the trade stream.

Two complementary detectors are combined:

1. RollingZScoreDetector — per-symbol, per-feature (price return, volume)
   rolling mean/std computed with an incremental (Welford-style) update, so
   memory and CPU stay O(1) per trade regardless of how long the stream has
   been running. Fast, interpretable, and a very standard first line of
   defense in real surveillance systems: "this is N standard deviations
   from recent normal behavior".

2. IsolationForestDetector — a scikit-learn IsolationForest refit
   periodically on a sliding window of recent multivariate feature vectors
   (return, volume, trade inter-arrival time) per symbol. Catches
   combinations of features that look unusual jointly even if no single
   feature crosses a hard z-score threshold — closer to how a real
   surveillance model would generalize beyond a fixed rule.

A trade is flagged if either detector fires; the resulting alert records
which detector(s) fired and why, since that's what an analyst reviewing the
alert would actually want to see.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    from sklearn.ensemble import IsolationForest
    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover - sklearn is in requirements.txt
    _HAS_SKLEARN = False


@dataclass
class Alert:
    trade_id: str
    symbol: str
    timestamp: float
    price: float
    volume: int
    reasons: list[str] = field(default_factory=list)
    scores: dict = field(default_factory=dict)
    # Ground truth passthrough, purely for the demo/eval script.
    is_injected_anomaly: bool = False
    anomaly_kind: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "price": round(self.price, 4),
            "volume": self.volume,
            "reasons": self.reasons,
            "scores": {k: round(v, 3) for k, v in self.scores.items()},
            "is_injected_anomaly": self.is_injected_anomaly,
            "anomaly_kind": self.anomaly_kind,
        }


class _WelfordStats:
    """Incremental mean/variance so we never need to store full history."""

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, x: float):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

    @property
    def std(self) -> float:
        if self.n < 2:
            return 0.0
        return math.sqrt(self.m2 / (self.n - 1))

    def zscore(self, x: float) -> float:
        s = self.std
        if s < 1e-9:
            return 0.0
        return (x - self.mean) / s


class RollingZScoreDetector:
    """Per-symbol rolling z-score over price returns and trade volume."""

    def __init__(self, z_threshold: float = 4.0, warmup_trades: int = 30):
        self.z_threshold = z_threshold
        self.warmup_trades = warmup_trades
        self._last_price: dict[str, float] = {}
        self._return_stats: dict[str, _WelfordStats] = {}
        self._volume_stats: dict[str, _WelfordStats] = {}

    def evaluate(self, trade) -> tuple[list[str], dict]:
        symbol = trade.symbol
        ret_stats = self._return_stats.setdefault(symbol, _WelfordStats())
        vol_stats = self._volume_stats.setdefault(symbol, _WelfordStats())

        reasons: list[str] = []
        scores: dict = {}

        last_price = self._last_price.get(symbol)
        if last_price is not None and last_price > 0:
            ret = (trade.price - last_price) / last_price
            if ret_stats.n >= self.warmup_trades:
                z = ret_stats.zscore(ret)
                scores["return_zscore"] = z
                if abs(z) >= self.z_threshold:
                    reasons.append(
                        f"price return z-score {z:.2f} exceeds +/-{self.z_threshold}"
                    )
            ret_stats.update(ret)
        self._last_price[symbol] = trade.price

        if vol_stats.n >= self.warmup_trades:
            zv = vol_stats.zscore(trade.volume)
            scores["volume_zscore"] = zv
            if zv >= self.z_threshold:
                reasons.append(
                    f"volume z-score {zv:.2f} exceeds +{self.z_threshold}"
                )
        vol_stats.update(trade.volume)

        return reasons, scores


class IsolationForestDetector:
    """
    Per-symbol IsolationForest over [return, volume, inter-arrival time],
    refit periodically on a sliding window. Falls back to a no-op if
    scikit-learn isn't installed.
    """

    def __init__(
        self,
        window_size: int = 300,
        refit_every: int = 50,
        contamination: float = 0.02,
    ):
        self.window_size = window_size
        self.refit_every = refit_every
        self.contamination = contamination
        self._windows: dict[str, deque] = {}
        self._models: dict[str, "IsolationForest"] = {}
        self._since_refit: dict[str, int] = {}
        self._last_price: dict[str, float] = {}
        self._last_ts: dict[str, float] = {}

    def evaluate(self, trade) -> tuple[list[str], dict]:
        if not _HAS_SKLEARN:
            return [], {}

        symbol = trade.symbol
        window = self._windows.setdefault(symbol, deque(maxlen=self.window_size))

        last_price = self._last_price.get(symbol, trade.price)
        last_ts = self._last_ts.get(symbol, trade.timestamp)
        ret = (trade.price - last_price) / last_price if last_price else 0.0
        inter_arrival = max(0.0, trade.timestamp - last_ts)

        self._last_price[symbol] = trade.price
        self._last_ts[symbol] = trade.timestamp

        features = [ret, float(trade.volume), inter_arrival]
        window.append(features)

        reasons: list[str] = []
        scores: dict = {}

        if len(window) < max(50, self.refit_every):
            return reasons, scores  # not enough history yet

        since = self._since_refit.get(symbol, 0)
        model = self._models.get(symbol)
        if model is None or since >= self.refit_every:
            X = np.array(window)
            model = IsolationForest(
                n_estimators=100,
                contamination=self.contamination,
                random_state=42,
            )
            model.fit(X)
            self._models[symbol] = model
            self._since_refit[symbol] = 0
        else:
            self._since_refit[symbol] = since + 1

        score = model.decision_function([features])[0]  # higher = more normal
        pred = model.predict([features])[0]  # -1 = outlier, 1 = inlier
        scores["isolation_forest_score"] = float(score)
        if pred == -1:
            reasons.append(f"isolation forest flagged joint outlier (score={score:.3f})")

        return reasons, scores


class AnomalyPipelineDetector:
    """Combines both detectors into a single evaluate() call."""

    def __init__(
        self,
        z_threshold: float = 4.0,
        iforest_window: int = 300,
        iforest_refit_every: int = 50,
    ):
        self.zscore = RollingZScoreDetector(z_threshold=z_threshold)
        self.iforest = IsolationForestDetector(
            window_size=iforest_window, refit_every=iforest_refit_every
        )

    def evaluate(self, trade) -> Optional[Alert]:
        z_reasons, z_scores = self.zscore.evaluate(trade)
        f_reasons, f_scores = self.iforest.evaluate(trade)

        reasons = z_reasons + f_reasons
        if not reasons:
            return None

        scores = {**z_scores, **f_scores}
        return Alert(
            trade_id=trade.trade_id,
            symbol=trade.symbol,
            timestamp=trade.timestamp,
            price=trade.price,
            volume=trade.volume,
            reasons=reasons,
            scores=scores,
            is_injected_anomaly=trade.is_injected_anomaly,
            anomaly_kind=trade.anomaly_kind,
        )
