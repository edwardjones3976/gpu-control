## Summary

- 

## ACCEPT item (if applicable)

- [ ] 1 Read-only checks
- [ ] 2 Ollama control
- [ ] 3 Job runner + lease
- [ ] 4 Overnight scheduler (source only)

## Cursor review required

- [ ] Requested Cursor PASS/HOLD/FAIL before merge
- [ ] No direct `main` push
- [ ] No secrets / venv / models / `.ps1` / live Task Scheduler mutation

## Test plan

- [ ] `python -I local_gpu.py --selftest` (or Desk Code equivalent)
- [ ] Syntax/lint CI green
