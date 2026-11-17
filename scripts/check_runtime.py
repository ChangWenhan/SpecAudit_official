#!/usr/bin/env python3
"""Print versions relevant to local generation and vLLM judge deployment."""

from __future__ import annotations

import importlib
import importlib.util
import sys


def main() -> None:
    print("executable", sys.executable)
    print("python", sys.version.split()[0])
    for name in ("torch", "transformers", "vllm", "accelerate", "safetensors", "tokenizers"):
        if importlib.util.find_spec(name) is None:
            print(name, "MISSING")
        else:
            module = importlib.import_module(name)
            print(name, getattr(module, "__version__", "unknown"), getattr(module, "__file__", ""))
    if importlib.util.find_spec("torch") is not None:
        torch = importlib.import_module("torch")
        print("cuda_available", torch.cuda.is_available())
        print("cuda_devices", torch.cuda.device_count())


if __name__ == "__main__":
    main()
