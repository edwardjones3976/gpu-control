"""cuda_probe: quick, read-only check that PyTorch can use the local GPU, plus three measurements.

Run through Desk Code shell.run (one command line):
  C:\\Users\\edwar\\AppData\\Local\\Python\\pythoncore-3.14-64\\python.exe -I
      C:\\Users\\edwar\\Tools\\gpu\\cuda_probe.py
Prints one JSON object. Uses about 0.6 GB of GPU memory for a few seconds, then frees it.
"""
import json
import time

import torch

out = {"torch": torch.__version__, "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available()}
if out["cuda_available"]:
    dev = torch.device("cuda:0")
    p = torch.cuda.get_device_properties(dev)
    free, total = torch.cuda.mem_get_info(dev)
    out.update({
        "device": p.name,
        "capability": "%d.%d" % (p.major, p.minor),
        "arch_list": torch.cuda.get_arch_list(),
        "sm_120_native": "sm_120" in torch.cuda.get_arch_list(),
        "vram_total_gb": round(total / 2**30, 2),
        "vram_free_gb": round(free / 2**30, 2),
    })

    def matmul_tflops(dtype, n=8192, reps=10):
        a = torch.randn(n, n, device=dev, dtype=dtype)
        b = torch.randn(n, n, device=dev, dtype=dtype)
        torch.matmul(a, b)  # warm-up (and any kernel JIT)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(reps):
            torch.matmul(a, b)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t) / reps
        del a, b
        return round(2 * n**3 / dt / 1e12, 1)

    torch.backends.cuda.matmul.allow_tf32 = True
    out["tflops"] = {
        "bf16": matmul_tflops(torch.bfloat16),
        "fp16": matmul_tflops(torch.float16),
        "fp32_tf32": matmul_tflops(torch.float32, n=4096),
    }

    # Host-to-device copy speed from pinned memory: this is what the PCIe link limits.
    host = torch.empty(256 * 2**20, dtype=torch.uint8).pin_memory()  # 256 MiB
    devbuf = torch.empty_like(host, device=dev)
    devbuf.copy_(host, non_blocking=True)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(8):
        devbuf.copy_(host, non_blocking=True)
    torch.cuda.synchronize()
    out["h2d_gb_per_s"] = round(8 * host.numel() / (time.perf_counter() - t) / 1e9, 1)
    t = time.perf_counter()
    for _ in range(8):
        host.copy_(devbuf, non_blocking=True)
    torch.cuda.synchronize()
    out["d2h_gb_per_s"] = round(8 * host.numel() / (time.perf_counter() - t) / 1e9, 1)
    del host, devbuf
    torch.cuda.empty_cache()
    out["peak_alloc_gb"] = round(torch.cuda.max_memory_allocated(dev) / 2**30, 2)
print(json.dumps(out))
