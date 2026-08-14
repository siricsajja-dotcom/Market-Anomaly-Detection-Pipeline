# market-anomaly-detection - July 2026

A real-time market surveillance pipeline: a simulated trade feed streams
through a combined statistical + ML anomaly detector, and flagged events
show up live on a dashboard. Built independently as a way to dig into
market surveillance and real-time alerting hands-on, end to end.

![status](https://img.shields.io/badge/status-demo-blue)

## Quickstart

```bash
./run.sh
# open http://localhost:5050
```

(`run.sh` creates a venv, installs `requirements.txt`, and starts the
server. No external market data or API keys needed — the feed is
self-contained.)

Within a few seconds you should see the ticker tape start moving, the
symbol table update, and — every so often — a flagged alert appear on the
right as an anomaly is injected and caught.

## What it does

```
generator.py  →  detector.py  →  pipeline.py  →  server.py (Flask + SSE)  →  dashboard
 (simulated       (z-score +      (background       (live push to
  trade feed)      isolation       thread, history    browser)
                   forest)         buffers)
```

1. **`app/generator.py`** simulates a live trade feed for five symbols
   using a discretized geometric-Brownian-motion price process. It also
   deliberately injects anomalies — price spikes/crashes, volume bursts,
   and quote-stuffing-style micro-bursts — at a configurable rate. Because
   the generator *knows* which trades it injected, the project can report
   real precision/recall numbers (see `eval_detector.py`), not just "looks
   reasonable."

2. **`app/detector.py`** combines two detectors, and a trade is flagged if
   either fires:
   - **Rolling z-score** on price returns and volume, per symbol, computed
     incrementally (Welford's algorithm) so memory/CPU stay constant
     regardless of stream length. This is the standard first line of
     defense in real surveillance systems — "N standard deviations from
     recent normal" — and it's fully interpretable, which matters when an
     analyst has to explain *why* something got flagged.
   - **IsolationForest** (scikit-learn), refit periodically on a sliding
     window of `[price return, volume, inter-arrival time]` per symbol.
     Catches combinations of features that look unusual jointly even when
     no single feature crosses a hard threshold.

3. **`app/pipeline.py`** runs the generator + detector on a background
   thread, keeps bounded in-memory history for the dashboard, optionally
   logs every trade + flag decision to CSV, and fans out live events to
   however many dashboard tabs are connected via a simple pub/sub queue.

4. **`app/server.py`** is a small Flask app. Live updates use
   **Server-Sent Events** rather than websockets — one dependency, works
   with the plain dev server, and a browser's built-in `EventSource` is
   all the client needs.

5. **`static/index.html`** is the dashboard: a live ticker tape, a
   per-symbol stats table, and a scrolling feed of flagged alerts with the
   reasons each one fired.

## Evaluating detector quality offline

```bash
python3 eval_detector.py --trades 5000 --anomaly-prob 0.02
```

Runs a batch of simulated trades (no real-time pacing, no server) through
the same detector and reports precision/recall against the known-injected
anomalies, broken down by anomaly type — useful for tuning the z-score
threshold or the IsolationForest's `contamination` parameter. On a typical
run, sharp price spikes/crashes and outright volume bursts are caught at
close to 90-100% recall; the subtler `micro_burst` (quote-stuffing) pattern
is caught far less often per-trade, since it's a *rate* anomaly rather than
a single-trade outlier — a good example of where a per-trade detector hits
its ceiling.

## Known limitations / natural next steps

Worth being upfront about, since these are exactly what I'd want to talk
through in an interview:

- **Rate-based anomalies (e.g. quote stuffing) are under-detected** because
  the detector scores individual trades, not trade *arrival rate* per
  symbol. Adding a rolling trades-per-second counter per symbol would
  close this gap.
- **IsolationForest refits synchronously** on the pipeline thread every
  `refit_every` trades per symbol; fine for a demo, but a production
  system would refit asynchronously so a slow fit never backs up the
  ingest path.
- **No persistence beyond the CSV log** — history is an in-memory ring
  buffer, so it resets on restart. A real system would write to a
  time-series store (e.g. TimescaleDB/ClickHouse/Kafka) and let the
  dashboard query that instead of holding state in the process.
- **Single-process** — the generator, detector, and server all run in one
  Python process for simplicity. The natural evolution is to split
  ingestion (a real feed handler or Kafka consumer) from detection
  (a scoring service) from serving (the dashboard), so each scales
  independently.

## Project layout

```
app/
  generator.py   — simulated trade feed with injected ground-truth anomalies
  detector.py     — rolling z-score + IsolationForest detectors
  pipeline.py      — background thread, history buffers, pub/sub for SSE
  server.py         — Flask app: dashboard, /stream (SSE), /api/snapshot
static/index.html — dashboard UI
eval_detector.py  — offline precision/recall evaluation
tests/            — unit tests for the detector
```
