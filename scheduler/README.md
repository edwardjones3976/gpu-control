# Overnight scheduler (ACCEPT item 4)

Window: **1–7 AM America/Los_Angeles**. Task name: `GPU-Overnight-Queue`.

- `overnight_queue.py`: the task's entry point. Outside the window it exits at once. Inside, it loads the installed `Tools/gpu/jobs/runner.py` and runs `drain()` with a 07:00 deadline. Jobs near the deadline get the stop file, save progress and stay queued, and Warden sessions come first. Each run writes a summary line to `Tools/gpu/scheduler/runs.jsonl`.
- `GPU-Overnight-Queue.xml`: the task definition. Daily trigger at 01:00, 6-hour hard limit, least privilege, runs only while you're signed in. No catch-up runs, no waking the PC.
- `install_task.py`: `check` (validates the XML, no side effects), `install` (`schtasks /Create /XML` from a UTF-16 copy in scratch) and `status` (read-only query).

## Notices

A scheduled task has no Agent Room or Slack access. Failures and yields therefore go to `Tools/gpu/jobs/notices.jsonl`. The first peer active after 07:00 posts that night's notices once to `#director-admin-ops`.

## Order

The live task is registered **only after Cursor PASS** on the PR containing this exact XML: `python -I scheduler\install_task.py install` through Desk Code. Nothing in a PR registers or changes the live task.
