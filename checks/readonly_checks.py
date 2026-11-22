"""Read-only GPU / host checks (stubs for ACCEPT item 1).

Extend surgically in a PR: processes, free memory, Windows error/crash log tails,
HKCU\\Environment only. No firewall/drivers/BIOS/password work.
"""
from __future__ import annotations

import json
import sys
import urllib.request
from typing import Any


def ollama_tags(endpoint: str = "http://127.0.0.1:11434", timeout: float = 5.0) -> dict[str, Any]:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(endpoint + "/api/tags")
    try:
        with opener.open(req, timeout=timeout) as r:
            return {"ok": True, "body": json.loads(r.read().decode())}
    except Exception as e:  # noqa: BLE001 — report, never crash the checker
        return {"ok": False, "error": "%s: %s" % (type(e).__name__, e)}


def local_gpu_status() -> dict[str, Any]:
    sys.path.append(r"C:\Users\edwar\Tools\gpu")
    try:
        from local_gpu import tool_gpu_status  # type: ignore
        return {"ok": True, "status": tool_gpu_status()}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": "%s: %s" % (type(e).__name__, e)}


def main() -> int:
    out = {
        "ollama_tags": ollama_tags(),
        "gpu_status": local_gpu_status(),
        "notes": "stub — add process/memory/event-log checks in item-1 PR",
    }
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
