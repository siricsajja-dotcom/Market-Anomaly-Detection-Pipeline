"""
Simulated trade stream generator.

Produces a realistic-ish stream of individual trade ticks for a handful of
symbols using geometric Brownian motion for price, plus a Poisson-ish
process for trade arrival and volume. On top of that "normal" process, it
occasionally injects deliberate anomalies (price spikes, volume bursts,
quote-stuffing-style bursts of tiny trades) so the detection pipeline has
real signal to catch — and so we know ground truth for the injected events,
which makes it possible to report precision/recall in a demo/eval script.

This stands in for a real market data feed (e.g. a FIX/ITCH feed or a
vendor websocket) in a way that's fully self-contained and requires no
external data or network access.
"""
from __future__ import annotations

import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Iterator, Optional


@dataclass
class Trade:
    trade_id: str
    symbol: str
    timestamp: float
    price: float
    volume: int
    # Ground truth, only known because *we* injected it — the detector never
    # sees this field. Useful for measuring precision/recall in demos.
    is_injected_anomaly: bool = False
    anomaly_kind: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "price": round(self.price, 4),
            "volume": self.volume,
            "is_injected_anomaly": self.is_injected_anomaly,
            "anomaly_kind": self.anomaly_kind,
        }


@dataclass
class SymbolState:
    symbol: str
    price: float
    drift: float = 0.0
    vol: float = 0.15          # annualized-ish volatility factor driving the random walk
    base_volume: int = 200


class TradeGenerator:
    """Yields Trade objects at wall-clock intervals to simulate a live feed."""

    def __init__(
        self,
        symbols: Optional[list[str]] = None,
        ticks_per_second: float = 8.0,
        anomaly_probability: float = 0.015,
        seed: Optional[int] = None,
    ):
        self.symbols = symbols or ["ACME", "GLOBEX", "INITECH", "UMBRELLA", "STARK"]
        self.ticks_per_second = ticks_per_second
        self.anomaly_probability = anomaly_probability
        self._rng = random.Random(seed)
        self._state = {
            s: SymbolState(symbol=s, price=self._rng.uniform(20, 500))
            for s in self.symbols
        }
        self._burst_remaining: dict[str, int] = {s: 0 for s in self.symbols}

    def _normal_tick(self, st: SymbolState) -> Trade:
        # Simple discretized GBM-ish step for price, log-normal-ish volume.
        dt = 1.0 / (self.ticks_per_second * 60)  # pretend ticks_per_second*60 ticks ~ 1 "minute" of vol
        shock = self._rng.gauss(0, 1)
        ret = st.drift * dt + st.vol * (dt ** 0.5) * shock
        st.price = max(0.01, st.price * (1 + ret))
        volume = max(1, int(self._rng.gauss(st.base_volume, st.base_volume * 0.3)))
        return Trade(
            trade_id=str(uuid.uuid4())[:8],
            symbol=st.symbol,
            timestamp=time.time(),
            price=st.price,
            volume=volume,
        )

    def _inject_anomaly(self, st: SymbolState) -> Trade:
        kind = self._rng.choice(["price_spike", "price_crash", "volume_burst", "micro_burst"])

        if kind == "price_spike":
            st.price *= self._rng.uniform(1.03, 1.12)
            volume = max(1, int(self._rng.gauss(st.base_volume, st.base_volume * 0.3)))
        elif kind == "price_crash":
            st.price *= self._rng.uniform(0.88, 0.97)
            volume = max(1, int(self._rng.gauss(st.base_volume, st.base_volume * 0.3)))
        elif kind == "volume_burst":
            volume = int(st.base_volume * self._rng.uniform(8, 20))
        else:  # micro_burst: kick off several tiny rapid-fire trades (quote-stuffing-ish)
            self._burst_remaining[st.symbol] = self._rng.randint(5, 10)
            volume = max(1, int(st.base_volume * 0.05))

        return Trade(
            trade_id=str(uuid.uuid4())[:8],
            symbol=st.symbol,
            timestamp=time.time(),
            price=max(0.01, st.price),
            volume=volume,
            is_injected_anomaly=True,
            anomaly_kind=kind,
        )

    def stream(self, paced: bool = True) -> Iterator[Trade]:
        """Infinite generator. When paced=True (default, used by the live
        pipeline) it sleeps between ticks to approximate ticks_per_second.
        Set paced=False for offline batch evaluation, where you want to
        rip through thousands of simulated trades as fast as possible."""
        interval = 1.0 / self.ticks_per_second
        while True:
            symbol = self._rng.choice(self.symbols)
            st = self._state[symbol]

            if self._burst_remaining[symbol] > 0:
                self._burst_remaining[symbol] -= 1
                trade = Trade(
                    trade_id=str(uuid.uuid4())[:8],
                    symbol=symbol,
                    timestamp=time.time(),
                    price=st.price,
                    volume=max(1, int(st.base_volume * 0.05)),
                    is_injected_anomaly=True,
                    anomaly_kind="micro_burst",
                )
            elif self._rng.random() < self.anomaly_probability:
                trade = self._inject_anomaly(st)
            else:
                trade = self._normal_tick(st)

            yield trade
            if paced:
                time.sleep(interval)
