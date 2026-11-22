"""GPU job runner stub — queue reviewed jobs under the local_gpu lease.

Runtime queues/logs live under Tools/gpu/jobs/ (gitignored). This module is source only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

RUNTIME_ROOT = Path(r"C:\Users\edwar\Tools\gpu\jobs")


def list_pending(queue_dir: Path | None = None) -> list[Path]:
    root = queue_dir or (RUNTIME_ROOT / "queue")
    if not root.is_dir():
        return []
    return sorted(p for p in root.glob("*.json") if p.is_file())


def load_job(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "label" not in data:
        raise ValueError("job must be an object with at least label")
    return data


def run_job_stub(job: dict[str, Any], provider: str = "unknown") -> dict[str, Any]:
    """Admit via lease then no-op. Real runners land in later PRs."""
    sys.path.append(r"C:\Users\edwar\Tools\gpu")
    from local_gpu import lease  # type: ignore

    vram = float(job.get("vram_gb", 0.0))
    ram = float(job.get("ram_gb", 0.0))
    minutes = int(job.get("max_minutes", 30))
    with lease(job["label"], vram_gb=vram, ram_gb=ram, max_minutes=minutes, provider=provider):
        return {"status": "stub-ok", "label": job["label"], "note": "no payload executed"}


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(json.dumps({"pending": [str(p) for p in list_pending()]}))
        return 0
    job = load_job(Path(argv[1]))
    print(json.dumps(run_job_stub(job, provider=str(job.get("provider", "unknown")))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
