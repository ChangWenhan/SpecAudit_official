#!/usr/bin/env python3
"""Minimal parallel ModelScope model downloader (stdlib only, resumable).

Usage: python3 ms_download.py <org/model> [<dest_root>]
"""

from __future__ import annotations

import json
import os
import ssl
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

FILES_API = "https://modelscope.cn/api/v1/models/{model}/repo/files?Revision=master&Recursive=true"
FILE_API = "https://modelscope.cn/api/v1/models/{model}/repo"


def ssl_context():
    # Some lab networks terminate TLS with a local proxy certificate.  This is
    # opt-in and used only for public model downloads on those hosts.
    if os.environ.get("MS_INSECURE_SSL") == "1":
        return ssl._create_unverified_context()
    return None


def list_files(model: str):
    with urllib.request.urlopen(
        FILES_API.format(model=model), timeout=60, context=ssl_context()
    ) as r:
        data = json.load(r)
    return [f for f in data["Data"]["Files"] if f.get("Type") == "blob"]


def file_url(model: str, path: str) -> str:
    q = urllib.parse.urlencode({"Revision": "master", "FilePath": path})
    return FILE_API.format(model=model) + "?" + q


def download_one(model: str, path: str, size: int, dest: str) -> str:
    target = os.path.join(dest, path)
    if os.path.exists(target) and os.path.getsize(target) == size:
        return f"skip   {path}"
    os.makedirs(os.path.dirname(target) or dest, exist_ok=True)
    tmp = target + ".part"
    req = urllib.request.Request(file_url(model, path), headers={"User-Agent": "ms-dl/1.0"})
    with urllib.request.urlopen(req, timeout=120, context=ssl_context()) as resp, open(tmp, "wb") as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, target)
    return f"done   {path} ({size/1e6:.1f} MB)"


def main():
    model = sys.argv[1]
    root = sys.argv[2] if len(sys.argv) > 2 else "models"
    name = model.split("/")[-1]
    dest = os.path.join(root, name)
    files = list_files(model)
    total = sum(f["Size"] for f in files)
    print(f"[{name}] {len(files)} files, {total/1e9:.2f} GB -> {dest}", flush=True)
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {
            ex.submit(download_one, model, f["Path"], f["Size"], dest): f["Path"]
            for f in files
        }
        for fut in as_completed(futs):
            try:
                print(fut.result(), flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"ERROR  {futs[fut]}: {e}", flush=True)


if __name__ == "__main__":
    main()
