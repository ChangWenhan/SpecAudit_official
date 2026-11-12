"""Integration checks for pool resume, frozen inputs and final-only decisions."""
import hashlib
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class PoolIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.requests = []
        calls = self.requests

        class Handler(BaseHTTPRequestHandler):
            def send(self, obj):
                body = json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                self.send({"data": [{"id": "fake-gptoss"}, {"id": "fake-guard"}]})

            def do_POST(self):
                obj = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                calls.append(obj)
                is_guard = obj["model"] == "fake-guard"
                text = " ".join(m["content"] for m in obj["messages"])
                if "analysis-only-case" in text and not (
                        getattr(self.server, "repair_mode", False) and not is_guard):
                    message = {"content": None, "reasoning_content": '{"violation": 1}'}
                elif is_guard:
                    label = "Controversial" if "controversial-case" in text else "Safe"
                    message = {"content": f"Safety: {label}\nCategories: None\nRefusal: No"}
                else:
                    value = 1 if "unsafe-case" in text else 0
                    message = {"content": json.dumps({"violation": value, "policy_category": None})}
                self.send({"model": obj["model"],
                           "choices": [{"message": message, "finish_reason": "stop"}],
                           "usage": {"completion_tokens": 8}})

            def log_message(self, *_):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.source = self.root / "fixture.jsonl"
        self.rows = [{"id": str(i), "sample_idx": 0, "prompt": prompt, "response": "fixture answer",
                      "dataset": "fixture", "mode": "fixture"} for i, prompt in enumerate(
                          ["safe-case", "unsafe-case", "controversial-case", "analysis-only-case"])]
        self.source.write_text("".join(json.dumps(r) + "\n" for r in self.rows))
        entry = {"file": self.source.name, "sha256": hashlib.sha256(self.source.read_bytes()).hexdigest(),
                 "records": len(self.rows), "dataset": "fixture", "mode": "fixture", "reuse_main": False}
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(json.dumps({"cells": [entry]}))
        url = f"http://127.0.0.1:{self.server.server_port}/v1/chat/completions"
        config = {"settings": {"gptoss_budgets": [16, 32], "timeout": 3, "seed": 0},
                  "endpoints": [{"name": f"{judge}-{i}", "judge": judge, "workers": 2,
                                 "url": url, "model": "fake-" + judge}
                                for judge in ("gptoss", "guard") for i in range(2)]}
        self.config = self.root / "config.json"
        self.config.write_text(json.dumps(config))
        self.status = self.root / "status.json"
        self.cmd = [sys.executable, "-m", "scripts.run_dual_judge_pool",
                    "--manifest", str(self.manifest), "--config", str(self.config),
                    "--checkpoint", str(self.root / "checkpoint.sqlite3"),
                    "--status", str(self.status)]

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)
        self.tmp.cleanup()

    def run_pool(self, *flags):
        return subprocess.run(self.cmd + list(flags), cwd=ROOT, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=45)

    def test_export_resume_and_unknown_retry(self):
        initial = self.run_pool()
        self.assertEqual(initial.returncode, 0, initial.stderr)
        self.assertEqual(json.loads(self.status.read_text())["labels_saved"], 8)
        judges = {}
        for judge, field in (("gptoss", "guard_gptoss"), ("guard_strong", "guard_strong")):
            output = self.source.with_suffix("." + judge + ".jsonl")
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual([r["id"] for r in rows], [r["id"] for r in self.rows])
            self.assertEqual(rows[-1][field]["safety"], "unknown")
            self.assertEqual(rows[-1]["response"], "fixture answer")
            judges[judge] = rows
        self.assertEqual(judges["guard_strong"][2]["guard_strong"]["safety"], "controversial")
        self.assertEqual(judges["gptoss"][1]["guard_gptoss"]["safety"], "unsafe")
        before = len(self.requests)
        resumed = self.run_pool()
        self.assertEqual(resumed.returncode, 0, resumed.stderr)
        self.assertEqual(len(self.requests), before, "resume must not rejudge saved labels")
        retry = self.run_pool("--retry-unknown")
        self.assertEqual(retry.returncode, 0, retry.stderr)
        self.assertEqual(len(self.requests) - before, 3, "retry only the two unknown samples")
        for payload in self.requests:
            if payload["model"] == "fake-gptoss":
                self.assertEqual(payload["reasoning_effort"], "low")
                self.assertEqual(payload["chat_template_kwargs"], {"reasoning_effort": "low"})
            else:
                self.assertFalse(payload["chat_template_kwargs"]["add_generation_prompt"])

    def test_changed_frozen_input_aborts_before_api(self):
        self.source.write_text(self.source.read_text().replace("fixture answer", "changed answer"))
        result = self.run_pool()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("frozen source changed", result.stderr)
        self.assertEqual(self.requests, [])

    def test_single_judge_phase_and_resume(self):
        initial = self.run_pool("--judges", "gptoss")
        self.assertEqual(initial.returncode, 0, initial.stderr)
        status = json.loads(self.status.read_text())
        self.assertEqual(status["labels_expected"], 4)
        self.assertEqual(status["labels_saved"], 4)
        self.assertTrue(all(p["model"] == "fake-gptoss" for p in self.requests))
        before = len(self.requests)
        resumed = self.run_pool("--judges", "gptoss")
        self.assertEqual(resumed.returncode, 0, resumed.stderr)
        self.assertEqual(len(self.requests), before)
        self.assertFalse(self.source.with_suffix(".guard_strong.jsonl").exists())

    def test_low_repair_changes_only_unknown_gptoss_and_preserves_guard(self):
        initial = self.run_pool()
        self.assertEqual(initial.returncode, 0, initial.stderr)
        gpt_path = self.source.with_suffix(".gptoss.jsonl")
        guard_path = self.source.with_suffix(".guard_strong.jsonl")
        before_rows = [json.loads(s) for s in gpt_path.read_text().splitlines()]
        guard_before = guard_path.read_bytes()
        self.server.repair_mode = True
        before_calls = len(self.requests)
        repair = subprocess.run(
            [sys.executable, "-m", "scripts.repair_dual_judge_unknown",
             "--config", str(self.config), "--manifest", str(self.manifest),
             "--runtime", str(self.root), "--budgets", "16", "32",
             "--workers-per-endpoint", "1"], cwd=ROOT, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=45)
        self.assertEqual(repair.returncode, 0, repair.stderr)
        self.assertEqual(len(self.requests) - before_calls, 1)
        self.assertEqual(self.requests[-1]["reasoning_effort"], "low")
        after_rows = [json.loads(s) for s in gpt_path.read_text().splitlines()]
        self.assertEqual(before_rows[:-1], after_rows[:-1])
        self.assertEqual(after_rows[-1]["guard_gptoss"]["safety"], "safe")
        self.assertEqual(after_rows[-1]["guard_gptoss"]["judge_run"]["reasoning_effort_location"], "top_level")
        self.assertEqual(guard_path.read_bytes(), guard_before)
        meta = json.loads(Path(str(gpt_path) + ".meta.json").read_text())
        self.assertTrue(meta["repairs"][0]["known_labels_preserved"])
        self.assertEqual(json.loads(self.status.read_text())["unknown_repair"]["resolved"], 1)
        self.assertEqual(len(list((self.root / "repairs").glob("*/before.tar.gz"))), 1)
        resumed = self.run_pool()
        self.assertEqual(resumed.returncode, 0, resumed.stderr)
        self.assertEqual(len(self.requests) - before_calls, 1)
        self.assertEqual(json.loads(Path(str(gpt_path) + ".meta.json").read_text())["repairs"], meta["repairs"])


if __name__ == "__main__":
    unittest.main()
