# Peer notes (gpu-control)

Identity context: peers coordinate as `provider@device` via Agent Room v3 (`http://127.0.0.1:8793/mcp`). Never address `zai@*`.

## Desk Code

- All coding / shell / `gh` side effects go through Desk Code (`room_code*`).
- Prefer `docker-python` / `docker-js`.
- Default shell cwd: `C:/Users/edwar/Tools/gpu/scratch`.
- PowerShell only via Desk Code `shell.run` (freeze elsewhere).
- `sealed.reveal` always needs human approve; other gated actions may use judgment receipts.

## Source vs runtime

- **Git source:** this repo (`C:/Users/edwar/gpu-control` → `https://github.com/edwardjones3976/gpu-control`).
- **Runtime:** `C:/Users/edwar/Tools/gpu` (venv, scratch, live ollama settings, job queues — **not** in git).
- **PR-only to main.** No direct main commits. Cursor PASS before merge.

## Sealed paths

Refuse walks into sealed folders. Do not open sealed files this task does not name.

## Review

Cursor reviews PASS/HOLD/FAIL surgically. DENY: PS outside Desk Code, firewall/drivers/BIOS/passwords, deletes outside scratch, Desk Code judgment edits without Cursor PASS.
