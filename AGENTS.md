# AGENTS.md

## Repository modernization directive (2026-focused)

When a task asks for large-scale Python refactoring, use this workflow.

### 1) Scope and planning
- Treat refactors as **incremental** and **safe-by-default**; do not rewrite every file in one step.
- Start with hotspots in backend compute paths (`comfy/`, `comfy_extras/`, `server.py`, `main.py`).
- Record a short plan before edits: target modules, expected gains, and fallback behavior.

### 2) Dependency alignment
- Keep dependency updates aligned with `requirements.txt`, `requirements-observability.txt`, and `pyproject.toml`.
- If a task requests newer runtimes (e.g. 2026 stack), prefer:
  - `numpy` for vectorized CPU paths,
  - `numba` for selective JIT kernels,
  - `pyarrow` for columnar/dataset I/O where it replaces Python loops,
  - optional `uvloop` for non-Windows asyncio runtime.
- If a request mentions a specific Torch build (e.g. "torch nightly131"), treat it as **optional/experimental** unless repository maintainers explicitly require it.

### 3) Performance refactor priorities
- Replace Python nested loops with vectorized NumPy/Torch first.
- Add Numba only for pure-NumPy kernels with measurable wins.
- Always keep a non-Numba fallback path.
- Prefer memory-stable operations (avoid unnecessary copies/casts).

### 4) Async/runtime priorities
- Prefer `asyncio` best practices: bounded queues, cancellation-safe awaits, and explicit timeouts.
- Enable `uvloop` only when available and platform-compatible (non-Windows).
- Never make uvloop a hard dependency unless specifically requested.

### 5) Typing/import cleanup
- Standardize typing imports from `typing` where possible.
- If you see `ty.astrall` / `ty.astral` mentions in requests, treat them as likely references to Astral tooling; verify exact package/tool names before changing dependencies.
- Reduce optional import side effects at module import time.
- Keep compatibility with existing public APIs and node interfaces.

### 6) Validation requirements
- Minimum checks after edits:
  - `python -m py_compile` for changed Python files,
  - focused runtime smoke tests for touched modules,
  - targeted unit tests when available.
- If environment misses heavy deps (torch/cuda), report limitations clearly and keep fallback logic covered by static checks.

### 7) Delivery expectations
- Include concise change summary + risks + follow-up opportunities.
- Prefer many small PRs over one monolithic refactor for "all .py files" requests.
