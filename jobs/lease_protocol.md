# Lease protocol

```python
import sys
sys.path.append(r"C:\Users\edwar\Tools\gpu")  # append, never insert
from local_gpu import lease

with lease("label", vram_gb=4, ram_gb=6, max_minutes=30, provider="claude"):
    ...
```

- Heavy if VRAM > 0.5 GB, RAM > 4 GB, or CPU workers > 8.
- Admission needs spare VRAM/RAM/commit; otherwise `Busy` / `Refused`.
- Windows job object + watchdog bound the process.
- Do not call Ollama outside the shared path during heavy work.
