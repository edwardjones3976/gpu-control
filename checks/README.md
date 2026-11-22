# Read-only checks

Stubs and imported probes for GPU/CUDA/Ollama/host memory. Prefer Desk Code `shell.run` with cwd under `Tools/gpu/scratch`.

| Script | Notes |
|--------|--------|
| `cuda_probe.py` | PyTorch CUDA probe (needs torch) |
| `jobmem_test.py` | Windows job memory vs CUDA commit |
| `readonly_checks.py` | Item 1. Sections: `processes` (top by commit), `memory`, `events` (System/Application critical+error), `crashes` (app crash/hang, Kernel-Power 41, BugCheck, display-driver reset 4101, unexpected shutdown, LiveKernelReports), `env` (`HKCU\Environment` only, allowlisted values; others show length only), `ollama`, `gpu`. `--section` repeatable, `--top`, `--events`, `--selftest`. |

Example: `python -I checks\readonly_checks.py --section memory --section crashes --events 5`

All sections are read-only. The only subprocess is `%SystemRoot%\System32\wevtutil.exe qe` on the System and Application logs. `LiveKernelReports` needs administrator rights, so under a normal user it reports `Access is denied`.
