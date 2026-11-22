# GPU job runner (ACCEPT item 3)

Runtime (not in git): `C:/Users/edwar/Tools/gpu/jobs/` with `queue/`, `done/`, `failed/`, `logs/`, `scripts/`, `approved.json`, `runs.jsonl` and `notices.jsonl`; the pinned environment is `C:/Users/edwar/Tools/gpu/venv/`.

`runner.py` subcommands: `list`, `validate JOB`, `run JOB`, `drain [--deadline ISO]`, `setup-venv`, `--selftest`.

## A job file (`queue/<name>.json`)

```json
{"label": "matmul-1", "script": "cuda_matmul.py", "sha256": "<64 hex>", "provider": "claude",
 "vram_gb": 1, "ram_gb": 2, "max_minutes": 10, "args": ["--size", "4096"]}
```

## Rules the runner enforces

- **Reviewed by hash.** The script must be a bare file name in `jobs/scripts/`. Its sha256 must match the file and be listed in `approved.json` with `reviewed_by` set to a provider other than `author` (same provider on another device counts as self). `approved.json` changes only through a reviewed PR.
- **One heavy job at a time.** Jobs run under `local_gpu.lease()`: VRAM, RAM and commit admission, a capped Windows job that the child process inherits, and `max_minutes`.
- **Warden first.** If a Warden session starts, or the window is about to end, the runner writes the job's stop file (`GPU_JOB_STOP_FILE`). The job saves progress in `GPU_JOB_CHECKPOINT_DIR` and exits; after 120 s the runner ends it. A yielded job stays queued and resumes from its checkpoint.
- **No retry.** A failed job moves to `failed/`, and one line goes to `notices.jsonl` for a peer to post.
- **Pinned packages.** `setup-venv` creates the venv on the base Python, which has the CUDA torch build, with `--system-site-packages`. It then installs `requirements-gpu.txt` with `--require-hashes --no-deps`.

`scripts/cuda_matmul.py` is a worked example of the stop-file and checkpoint protocol. Protocol details: [lease_protocol.md](lease_protocol.md).
