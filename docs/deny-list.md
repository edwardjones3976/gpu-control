# Deny-list (Cursor review)

Denied without separate owner/Cursor clearance:

1. PowerShell scripts (`.ps1`) or PS chains outside Desk Code `shell.run`
2. Firewall, driver, BIOS, or password changes
3. Deletes outside `Tools/gpu/scratch`
4. Desk Code judgment / policy edits without Cursor PASS
5. Committing secrets, `.env`, Ollama blobs, models, `.gguf`, venv, leases, wheels
6. Direct pushes to `main`
7. Live Task Scheduler mutations inside a PR (ship source only; register after PASS)
8. Addressing `zai@*`
9. Destroying work on corrections
