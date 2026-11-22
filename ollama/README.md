# Ollama control (ACCEPT item 2)

Runtime settings file (not in git): `C:/Users/edwar/Tools/gpu/ollama/settings.json`

- `control.py`: `status`, `plan`, `apply`, `rollback`, `pull`, `bench`, `--selftest`. Each prints one JSON line.
- `settings.schema.json`: shape of the settings file. `control.py` enforces the same rules with the standard library.
- `settings.example.json`: example values, with `allow_restart: false`.

## What each setting does

Each non-null key sets one user variable in `HKCU\Environment`; `null` leaves it alone.

| Key | Variable |
|---|---|
| `keep_alive` | `OLLAMA_KEEP_ALIVE` |
| `parallel` | `OLLAMA_NUM_PARALLEL` |
| `flash_attention` | `OLLAMA_FLASH_ATTENTION` (re-check T7 after changing) |
| `kv_cache_type` | `OLLAMA_KV_CACHE_TYPE` (`q8_0`/`q4_0` need flash attention, and re-check T7 after) |
| `max_loaded_models` | `OLLAMA_MAX_LOADED_MODELS` |
| `cuda_cache_maxsize` | `CUDA_CACHE_MAXSIZE` (compile cache for Ollama's PTX-only CUDA code) |

## Order (owner rules, 2026-11-21)

1. `bench --model local-warden:t7-ckpt153-q8 --levels 1,2,4 --load-twice` with today's settings. Measure first.
2. Write `Tools/gpu/ollama/settings.json` from the table, run `plan`, then `apply`. `apply` refuses while a Warden session runs or T7 is loaded. It saves the old variables to `previous.json`, writes the new ones, restarts Ollama outside the Desk Code job, and rolls back at once if GPU memory passes 11.5 GB or free PC memory drops under 2 GB.
3. `bench` again. On a breach, `rollback`, then notify.
4. Change `flash_attention` / `kv_cache_type` only with a T7 re-check (Cursor's 20-item test).

Every action appends one line to `Tools/gpu/ollama/control.log.jsonl`.
