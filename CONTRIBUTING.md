# Contributing to gpu-control

## Binding rule: PR-only to main

**No direct commits or pushes to `main`.** All work lands via pull request.

This amends Cursor ACCEPT seq **481** (`blackbox:1795318751355-78acba97`): Claude (and every peer) builds items 1→4 through PRs against this repository. Uncontrolled writes under `Tools/gpu` **source** paths outside a PR branch checkout are refused.

Branch protection on GitHub should enforce the same rule when the owner applies it; until then this document is **binding for Agent Room ACCEPT**.

## Branch naming

- `feat/<short>` — new capability
- `fix/<short>` — bug fix
- `chore/<short>` — docs/CI/scaffold
- `claude/<item>-<short>` — Claude ACCEPT items (1 checks, 2 ollama, 3 jobs, 4 scheduler)

## Review / merge

1. Open PR using the template (Cursor review required).
2. Cursor reviews **PASS / HOLD / FAIL** surgically.
3. Merge only after **Cursor PASS** (and any required CI).
4. Corrections stay surgical — never destroy on corrections.

## Allowlisted paths (source)

Peers may propose changes under:

- `local_gpu.py`
- `checks/`
- `ollama/` (schema + `settings.example.json` only — not live `settings.json`)
- `jobs/` (stubs, docs, allowlisted job templates — not `jobs/runtime/`)
- `scheduler/` (source XML/Python only — do not register the live task in the PR)
- `docs/`, `.github/`, top-level docs (`README`, `CONTRIBUTING`, `AGENTS`, `LICENSE`, dependency manifests)

## Out of scope / deny (see also docs/deny-list.md)

- PowerShell scripts or PS chains outside Desk Code
- Firewall / drivers / BIOS / passwords
- Deletes outside `scratch/`
- Desk Code judgment/policy edits without Cursor PASS
- Committing venv, models, `.gguf`, `.env`, leases, Ollama blobs, wheels

## Order of work (ACCEPT items)

1. More read-only checks
2. Ollama control → `Tools/gpu/ollama/settings.json` (runtime; source schema here)
3. GPU job runner + lease → `Tools/gpu/jobs/`, `Tools/gpu/venv/`
4. Overnight scheduler 1–7 AM PT → task `GPU-Overnight-Queue` (source here; live register only after PASS)

## Local runtime path

Default Desk Code cwd: `C:/Users/edwar/Tools/gpu/scratch`.  
Source checkout: `C:/Users/edwar/gpu-control`.
