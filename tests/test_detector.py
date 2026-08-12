import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.detector import RollingZScoreDetector
from app.generator import Trade


def make_trade(symbol, price, volume, ts=0.0):
    return Trade(trade_id="t", symbol=symbol, timestamp=ts, price=price, volume=volume)


def test_zscore_flags_large_price_jump():
    det = RollingZScoreDetector(z_threshold=3.0, warmup_trades=20)
    price = 100.0
    # Warm up with small, stable noise so std will stay low.
    for i in range(40):
        price *= 1 + (0.0005 if i % 2 == 0 else -0.0005)
        det.evaluate(make_trade("TEST", price, 100, ts=i))

    # A sudden 8% jump should stand out against that tight recent history.
    jump_price = price * 1.08
    reasons, scores = det.evaluate(make_trade("TEST", jump_price, 100, ts=999))
    assert any("return z-score" in r for r in reasons)
    assert abs(scores["return_zscore"]) >= 3.0


def test_zscore_does_not_flag_normal_noise():
    det = RollingZScoreDetector(z_threshold=4.0, warmup_trades=20)
    price = 50.0
    flagged_count = 0
    for i in range(200):
        price *= 1 + (0.001 if i % 2 == 0 else -0.001)
        reasons, _ = det.evaluate(make_trade("TEST", price, 100, ts=i))
        if reasons:
            flagged_count += 1
    # Consistent, small, alternating moves shouldn't trip a 4-sigma threshold.
    assert flagged_count == 0


def test_volume_burst_is_flagged():
    det = RollingZScoreDetector(z_threshold=3.0, warmup_trades=20)
    for i in range(40):
        det.evaluate(make_trade("TEST", 10.0, 100 + (i % 3), ts=i))
    reasons, scores = det.evaluate(make_trade("TEST", 10.0, 5000, ts=999))
    assert any("volume z-score" in r for r in reasons)
