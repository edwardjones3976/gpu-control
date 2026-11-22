# gpu-control

Shared **Providers → Desk Code → GPU** control plane for the BlackBox RTX 5070.

**North star:** the GPU enhances all peers (Claude, Codex, Cursor, Antigravity, …) — more ability, less wall time, fewer cloud tokens. It is a shared accelerator, not a replacement for cloud models as the main brain.

## Architecture

```
Providers (think / decide / coordinate)
        ↓  room_* · MCP · local_gpu tools
Desk Code (authorize · isolate · receipt)
        ↓  shell.run today · optional docker-gpu later
GPU / CUDA (Ollama · PyTorch · nvidia tools · local_gpu lease)
```

Plan (canonical HTML): `C:/Users/edwar/Agent-Roles/gpu-work-plan-20261121.html`

## Local layout

| Path | Role |
|------|------|
| **This repo** (`C:/Users/edwar/gpu-control`) | Source of truth; PRs only |
| `C:/Users/edwar/Tools/gpu` | Runtime install / Desk Code cwd parent |
| `C:/Users/edwar/Tools/gpu/scratch` | Default Desk Code shell cwd |
| `C:/Users/edwar/Tools/gpu/venv` | GPU venv (not in git) |
| `C:/Users/edwar/Tools/gpu/jobs` | Runtime job queue/logs (not in git) |

Install reviewed source into `Tools/gpu` after merge (copy or symlink). Do **not** commit venv, scratch, leases, models, or secrets.

## Package map

| Path | Purpose |
|------|---------|
| `local_gpu.py` | Lease + MCP tools (status, embed, rank, summarize, classify) |
| `checks/` | Read-only GPU/CUDA/Ollama/memory probes |
| `ollama/` | Settings schema + example (no live secrets) |
| `jobs/` | Job runner stubs + lease protocol |
| `scheduler/` | Overnight 1–7 AM PT task **source** (not a live schedule mutation) |
| `docs/` | North star, deny-list, plan link |

## How peers work

1. **Claude / others** open a branch, implement against allowlisted paths, open a **PR**.
2. **No direct pushes to `main`.**
3. **Cursor** reviews surgically (PASS / HOLD / FAIL) before merge.
4. Runtime installs land under `Tools/gpu` only after merge + Cursor PASS.

See [CONTRIBUTING.md](CONTRIBUTING.md) and [AGENTS.md](AGENTS.md).

## Quick checks (Desk Code)

```text
python -I C:\Users\edwar\Tools\gpu\local_gpu.py --selftest
python -I C:\Users\edwar\Tools\gpu\checks\cuda_probe.py
```

Torch/CUDA is optional for pure lease/MCP work; see `requirements.txt`.

## Safety

- No credentials in this repo.
- No PowerShell scripts (`.ps1`) — PowerShell freeze; PS only via Desk Code `shell.run` when authorized.
- Sealed-path refuse; peers-only Agent Room (`provider@device`; never `zai@*`).
