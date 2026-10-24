#!/usr/bin/env python3
"""Batched HTTP service using the project's existing Qwen3Guard classifier."""
from __future__ import annotations

import argparse
import json
import queue
import threading
import time
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from src.judge import Qwen3Guard


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-id", default="qwen3guard-4b")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--batch-wait-ms", type=int, default=25)
    args = ap.parse_args()
    guard = Qwen3Guard(args.model, "cuda")
    waiting = queue.Queue(maxsize=256)

    def batch_worker():
        while True:
            jobs = [waiting.get()]
            deadline = time.monotonic() + args.batch_wait_ms / 1000
            while len(jobs) < args.batch_size:
                timeout = deadline - time.monotonic()
                if timeout <= 0:
                    break
                try:
                    jobs.append(waiting.get(timeout=timeout))
                except queue.Empty:
                    break
            try:
                verdicts = guard.classify([pair for pair, future in jobs],
                                          batch_size=args.batch_size)
                for (_, future), verdict in zip(jobs, verdicts):
                    future.set_result(verdict)
            except Exception as exc:
                for _, future in jobs:
                    future.set_exception(exc)

    threading.Thread(target=batch_worker, daemon=True, name="guard-batch").start()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def send(self, status, obj):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.rstrip("/") not in ("/health", "/v1/models"):
                return self.send(404, {"error": "not found"})
            self.send(200, {"data": [{"id": args.model_id, "object": "model"}],
                            "batch_size": args.batch_size,
                            "classifier": "src.judge.Qwen3Guard"})

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                return self.send(404, {"error": "not found"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 1_000_000:
                    return self.send(400, {"error": "invalid request length"})
                obj = json.loads(self.rfile.read(length))
                messages = obj["messages"]
                if obj.get("model") != args.model_id or [m["role"] for m in messages] != ["user", "assistant"]:
                    return self.send(400, {"error": "expected judge model and user/assistant pair"})
                future = Future()
                waiting.put(((messages[0]["content"], messages[1]["content"]), future), timeout=30)
                verdict = future.result(timeout=900)
                self.send(200, {"model": args.model_id, "object": "chat.completion",
                                "choices": [{"index": 0, "finish_reason": "stop",
                                             "message": {"role": "assistant",
                                                         "content": verdict.get("raw") or "Safety: Unknown"}}]})
            except Exception as exc:
                self.send(500, {"error": {"message": str(exc)[:300]}})

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"READY {args.host}:{args.port} model={args.model_id} batch={args.batch_size}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
