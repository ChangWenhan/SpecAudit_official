"""Minimal OpenAI-compatible chat client.

Lets the TAP / Crescendo *attacker* (and any future remote judge) run as a
service instead of being loaded onto the local GPU.  The local GPU then only
has to hold the victim under test, so a single attacker deployment can serve
all three victim models.

Endpoint is expected to be an OpenAI-compatible `/v1/chat/completions` URL
(e.g. a vLLM server).  For assistant-prefill generation (TAP needs the
official `{"improvement": ...` prefill) the server must support
`continue_final_message` in `chat_template_kwargs`; only the continuation is
returned, so callers must prepend the prefill themselves.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import requests


class RemoteChat:
    def __init__(
        self,
        endpoint: str,
        model: Optional[str] = None,
        timeout: int = 300,
        api_key: Optional[str] = None,
        extra_kwargs: Optional[Dict] = None,
    ):
        self.endpoint = endpoint
        self.model = model or endpoint
        self.timeout = timeout
        self.api_key = api_key
        self.extra_kwargs = dict(extra_kwargs or {})

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new: int = 512,
        temperature: float = 0.0,
        top_p: Optional[float] = 1.0,
        seed: Optional[int] = None,
        prefill: Optional[str] = None,
        enable_thinking: Optional[bool] = None,
        continue_final: bool = False,
    ) -> str:
        msgs = list(messages)
        payload: Dict = {
            "model": self.model,
            "messages": msgs,
            "temperature": temperature,
            "max_tokens": max_new,
        }
        if top_p is not None:
            payload["top_p"] = top_p
        if seed is not None:
            payload["seed"] = seed

        ctk = dict(self.extra_kwargs)
        if prefill is not None:
            msgs = msgs + [{"role": "assistant", "content": prefill}]
            payload["messages"] = msgs
            ctk["continue_final_message"] = True
            ctk["add_generation_prompt"] = False
        elif continue_final:
            ctk["continue_final_message"] = True
            ctk["add_generation_prompt"] = False
        if enable_thinking is not None:
            ctk["enable_thinking"] = enable_thinking
        if ctk:
            payload["chat_template_kwargs"] = ctk

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        session = requests.Session()
        session.trust_env = False  # ignore shell http(s)_proxy for LAN endpoints
        r = session.post(self.endpoint, json=payload, headers=headers, timeout=self.timeout)
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]
        return msg.get("content") or msg.get("reasoning_content") or ""


class RemoteGuard:
    """Qwen3Guard served behind an OpenAI-compatible endpoint.

    Same interface as src.judge.Qwen3Guard.classify: takes [(prompt, response)]
    and returns the parsed Qwen3Guard verdicts.  The guard chat is rendered as
    [user, assistant] and the model continues the final assistant turn
    (continue_final_message) to emit the `Safety: ...` verdict, matching the
    local HF path in src/judge.py.
    """

    def __init__(self, endpoint: str, model: Optional[str] = None, timeout: int = 180,
                 workers: int = 8, api_key: Optional[str] = None):
        from .judge import parse_guard_output

        self._parse = parse_guard_output
        self.client = RemoteChat(endpoint, model, timeout=timeout, api_key=api_key)
        self.workers = workers

    def _one(self, p: str, r: str) -> Dict:
        if r is None or not str(r).strip():
            return {"safety": "safe", "refusal": "unknown", "categories": [],
                    "raw": "", "rule": "empty_response"}
        try:
            text = self.client.chat(
                [
                    {"role": "user", "content": "" if p is None else str(p)},
                    {"role": "assistant", "content": "" if r is None else str(r)},
                ],
                max_new=64,
                temperature=0.0,
                continue_final=True,
            )
        except Exception as e:  # noqa: BLE001
            return {"safety": "unknown", "refusal": "unknown", "categories": [], "raw": f"error: {e}"}
        return self._parse(text)

    def classify(self, pairs, batch_size: int = 8):
        from concurrent.futures import ThreadPoolExecutor

        pairs = list(pairs)
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            return list(ex.map(lambda pr: self._one(pr[0], pr[1]), pairs))
