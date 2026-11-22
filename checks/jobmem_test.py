"""jobmem_test: does a Windows Job Object memory cap (JOB_OBJECT_LIMIT_JOB_MEMORY) trip falsely on a CUDA job?

Run through Desk Code shell.run:
  python.exe -I C:\\Users\\edwar\\Tools\\gpu\\jobmem_test.py            measure only, no cap
  python.exe -I C:\\Users\\edwar\\Tools\\gpu\\jobmem_test.py <cap_gb>   put this process in a capped job first
Prints one JSON object: committed memory (PrivateUsage) after each CUDA stage, and where a capped run fails.
Uses up to about 2.5 GB of GPU memory for a few seconds. Refuses to start with under 6 GB of RAM available.
"""
import ctypes
import ctypes.wintypes as W
import json
import sys

GIB = 2**30
k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.GetCurrentProcess.restype = W.HANDLE
k32.CreateJobObjectW.restype = W.HANDLE
k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
k32.AssignProcessToJobObject.argtypes = [W.HANDLE, W.HANDLE]
k32.SetInformationJobObject.argtypes = [W.HANDLE, ctypes.c_int, ctypes.c_void_p, W.DWORD]
k32.QueryInformationJobObject.argtypes = [W.HANDLE, ctypes.c_int, ctypes.c_void_p, W.DWORD, ctypes.c_void_p]
k32.IsProcessInJob.argtypes = [W.HANDLE, W.HANDLE, ctypes.POINTER(W.BOOL)]
k32.K32GetProcessMemoryInfo.argtypes = [W.HANDLE, ctypes.c_void_p, W.DWORD]


class PMC(ctypes.Structure):  # PROCESS_MEMORY_COUNTERS_EX
    _fields_ = [("cb", W.DWORD), ("PageFaultCount", W.DWORD)] + [
        (n, ctypes.c_size_t)
        for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                  "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage",
                  "PrivateUsage")
    ]


class MEMSTAT(ctypes.Structure):  # MEMORYSTATUSEX
    _fields_ = [("dwLength", W.DWORD), ("dwMemoryLoad", W.DWORD)] + [
        (n, ctypes.c_ulonglong)
        for n in ("ullTotalPhys", "ullAvailPhys", "ullTotalPageFile", "ullAvailPageFile",
                  "ullTotalVirtual", "ullAvailVirtual", "ullAvailExtendedVirtual")
    ]


class BASIC(ctypes.Structure):  # JOBOBJECT_BASIC_LIMIT_INFORMATION
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", W.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", W.DWORD),
                ("Affinity", ctypes.c_size_t), ("PriorityClass", W.DWORD), ("SchedulingClass", W.DWORD)]


class EXT(ctypes.Structure):  # JOBOBJECT_EXTENDED_LIMIT_INFORMATION
    _fields_ = [("Basic", BASIC), ("IoInfo", ctypes.c_ulonglong * 6), ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t)]


EXTENDED_LIMIT_INFO = 9
LIMIT_JOB_MEMORY = 0x200
LIMIT_KILL_ON_JOB_CLOSE = 0x2000


def committed_gb():
    p = PMC()
    p.cb = ctypes.sizeof(p)
    k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(p), p.cb)
    return round(p.PrivateUsage / GIB, 3)


def avail():
    m = MEMSTAT()
    m.dwLength = ctypes.sizeof(m)
    k32.GlobalMemoryStatusEx(ctypes.byref(m))
    return round(m.ullAvailPhys / GIB, 2), round(m.ullAvailPageFile / GIB, 2)


out = {"cap_gb": float(sys.argv[1]) if len(sys.argv) > 1 else None}
inj = W.BOOL()
k32.IsProcessInJob(k32.GetCurrentProcess(), None, ctypes.byref(inj))
out["already_in_job"] = bool(inj.value)  # did shell.run put us in a job already?
out["ram_avail_gb"], out["commit_avail_gb"] = avail()
if out["ram_avail_gb"] < 6:
    out["refused"] = "under 6 GB of RAM available; nothing was run"
    print(json.dumps(out))
    sys.exit(0)

job = None
if out["cap_gb"] is not None:
    job = k32.CreateJobObjectW(None, None)
    info = EXT()
    info.Basic.LimitFlags = LIMIT_JOB_MEMORY | LIMIT_KILL_ON_JOB_CLOSE
    info.JobMemoryLimit = int(out["cap_gb"] * GIB)
    out["set_limit_ok"] = bool(k32.SetInformationJobObject(job, EXTENDED_LIMIT_INFO, ctypes.byref(info), ctypes.sizeof(info)))
    out["assign_ok"] = bool(k32.AssignProcessToJobObject(job, k32.GetCurrentProcess()))
    out["assign_error"] = ctypes.get_last_error() if not out["assign_ok"] else 0

stages = [{"stage": "baseline", "committed_gb": committed_gb()}]
out["stages"] = stages
try:
    import torch
    stages.append({"stage": "import torch", "committed_gb": committed_gb()})
    x = torch.ones(1, device="cuda")
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    stages.append({"stage": "cuda context", "committed_gb": committed_gb(), "vram_free_gb": round(free / GIB, 2)})
    n = int(min(2 * GIB, max(0, free - GIB)))  # leave 1 GB of VRAM for whatever else is loaded
    big = torch.empty(n, dtype=torch.uint8, device="cuda")
    big.fill_(1)
    torch.cuda.synchronize()
    stages.append({"stage": "alloc %.2f GB VRAM" % (n / GIB), "committed_gb": committed_gb()})
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    (a @ a).sum().item()  # first matmul loads cuBLAS and its workspace
    stages.append({"stage": "fp16 matmul (cuBLAS)", "committed_gb": committed_gb()})
    host = torch.empty(256 * 2**20, dtype=torch.uint8).pin_memory()
    big[: host.numel()].copy_(host)
    torch.cuda.synchronize()
    stages.append({"stage": "256 MiB pinned host copy", "committed_gb": committed_gb()})
    del big, a, host, x
    torch.cuda.empty_cache()
    stages.append({"stage": "freed", "committed_gb": committed_gb()})
    out["result"] = "completed"
except BaseException as e:  # a capped run is expected to fail somewhere; record where and how
    out["result"] = "failed"
    out["error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
    stages.append({"stage": "at failure", "committed_gb": committed_gb()})
if job:
    q = EXT()
    k32.QueryInformationJobObject(job, EXTENDED_LIMIT_INFO, ctypes.byref(q), ctypes.sizeof(q), None)
    out["peak_job_committed_gb"] = round(q.PeakJobMemoryUsed / GIB, 3)
    off = EXT()  # drop the limits before exit so closing the handle doesn't matter
    k32.SetInformationJobObject(job, EXTENDED_LIMIT_INFO, ctypes.byref(off), ctypes.sizeof(off))
print(json.dumps(out))
