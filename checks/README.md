# Read-only checks

Stubs and imported probes for GPU/CUDA/Ollama/host memory. Prefer Desk Code `shell.run` with cwd under `Tools/gpu/scratch`.

| Script | Notes |
|--------|--------|
| `cuda_probe.py` | PyTorch CUDA probe (needs torch) |
| `jobmem_test.py` | Windows job memory vs CUDA commit |
| `readonly_checks.py` | Process/memory/Ollama status stubs (extend in item 1) |
