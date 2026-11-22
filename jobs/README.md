# GPU job runner (source stubs)

Runtime (not in git):

- `C:/Users/edwar/Tools/gpu/jobs/` — queue, logs, runtime state
- `C:/Users/edwar/Tools/gpu/venv/` — pinned GPU environment

Protocol: [lease_protocol.md](lease_protocol.md)

Jobs must take a `local_gpu.lease(...)` before heavy CUDA/Ollama work. One heavy job at a time.
