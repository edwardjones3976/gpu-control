"""Example GPU job: timed fp16 matrix multiplies that honour the runner's stop file and checkpoint.

The runner sets GPU_JOB_STOP_FILE (exists = save progress and exit soon) and GPU_JOB_CHECKPOINT_DIR.
Usage (via jobs/runner.py only): cuda_matmul.py [--size N] [--rounds R]
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--size", type=int, default=4096)
ap.add_argument("--rounds", type=int, default=200)
a = ap.parse_args()
stop = Path(os.environ.get("GPU_JOB_STOP_FILE", "__none__"))
ckpt_dir = Path(os.environ.get("GPU_JOB_CHECKPOINT_DIR", "."))
ckpt_dir.mkdir(parents=True, exist_ok=True)
ckpt = ckpt_dir / "progress.json"
done = json.loads(ckpt.read_text())["done"] if ckpt.is_file() else 0  # resume after a yield

x = torch.randn(a.size, a.size, device="cuda", dtype=torch.float16)
torch.cuda.synchronize()
t0 = time.perf_counter()
for i in range(done, a.rounds):
    y = x @ x
    torch.cuda.synchronize()
    if stop.exists():
        ckpt.write_text(json.dumps({"done": i + 1}))
        print(json.dumps({"yielded_at_round": i + 1, "reason": stop.read_text()}))
        raise SystemExit(3)
elapsed = time.perf_counter() - t0
rounds = a.rounds - done
tflops = 2 * a.size**3 * rounds / elapsed / 1e12 if elapsed else None
ckpt.unlink(missing_ok=True)
print(json.dumps({"rounds": rounds, "seconds": round(elapsed, 3), "tflops_fp16": round(tflops, 1) if tflops else None,
                  "device": torch.cuda.get_device_name(0)}))
