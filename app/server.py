"""
Flask app exposing:
  GET  /                 -> dashboard (static/index.html)
  GET  /stream            -> Server-Sent Events live feed of trades + alerts
  GET  /api/snapshot       -> recent trade/alert history + running stats (initial page load)
  GET  /api/health          -> liveness check

Uses Server-Sent Events (plain HTTP, no extra dependency) rather than
websockets so the whole project only needs Flask — no ASGI server, no
extra client library, works with `flask run` or the built-in dev server.
"""
from __future__ import annotations

import json
import os
import time

from flask import Flask, Response, jsonify, send_from_directory

from .pipeline import AnomalyPipeline

STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="")

pipeline = AnomalyPipeline(
    ticks_per_second=float(os.environ.get("TICKS_PER_SECOND", 8.0)),
    anomaly_probability=float(os.environ.get("ANOMALY_PROBABILITY", 0.015)),
    log_path=os.environ.get("LOG_PATH", "data/trade_log.csv"),
)
pipeline.start()


@app.get("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "time": time.time()})


@app.get("/api/snapshot")
def snapshot():
    return jsonify(pipeline.snapshot())


@app.get("/stream")
def stream():
    q = pipeline.subscribe()

    def gen():
        try:
            # Send an initial snapshot so a freshly opened dashboard isn't empty.
            yield f"event: snapshot\ndata: {json.dumps(pipeline.snapshot())}\n\n"
            while True:
                try:
                    event = q.get(timeout=15)
                    yield f"event: tick\ndata: {json.dumps(event)}\n\n"
                except Exception:
                    yield ": keep-alive\n\n"  # SSE comment line to keep connection open
        finally:
            pipeline.unsubscribe(q)

    return Response(
        gen(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5050)), threaded=True)
