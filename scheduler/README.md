# Overnight scheduler (source only)

Window: **1–7 AM America/Los_Angeles**. Task name: `GPU-Overnight-Queue`.

This directory holds the **source** definition. Do **not** register or mutate the live scheduled task inside a PR. After Cursor PASS, an authorized Desk Code step may install it.

Files:

- `overnight_queue.py` — queue drain entrypoint (calls job runner under lease)
- `GPU-Overnight-Queue.xml` — Task Scheduler XML export template (source)
