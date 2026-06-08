"""Locked-down subprocess executor for UNTRUSTED candidate (model) programs.

Constraint (build spec §0.6, §4): candidate source runs in a restricted
subprocess with:
* no network (socket creation is disabled inside the child),
* no filesystem writes outside a per-run temp dir (deterministic open-guard),
* a wall-clock cap (parent kills the process group on timeout) AND a CPU cap
  (RLIMIT_CPU, so a busy loop is reaped even if it ignores wall-clock), and
* a memory cap (RLIMIT_AS).

This is defense-in-depth in pure Python, not a hardware sandbox; true isolation
would use containers/seccomp. It is sufficient for the threat model here
(our own and model-authored transform functions).

The executor is *batch-first*: ``run_candidate_batch`` compiles the candidate
once and maps ``transform`` over many inputs in a single subprocess (so scoring
2000 rows costs one process, not 2000). ``run_candidate`` is the single-row
wrapper.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from typing import Any

_SENTINEL = "@@PROGRECON_RESULT@@"

_DRIVER = r'''
import sys, json, io, builtins, tempfile, os, shutil

# Best-effort network block (in-process): neutralize socket creation.
import socket as _socket
def _blocked(*a, **k):
    raise OSError("network access is disabled in the sandbox")
_socket.socket = _blocked
_socket.create_connection = _blocked
try:
    _socket.create_server = _blocked
except Exception:
    pass

# Confine filesystem writes to a per-run temp dir; deny writes anywhere else.
_TMPDIR = tempfile.mkdtemp(prefix="prsbx_")
_real_open = builtins.open
def _guarded_open(file, mode="r", *a, **k):
    m = mode if isinstance(mode, str) else "r"
    if any(c in m for c in ("w", "a", "x", "+")):
        p = os.path.abspath(os.fspath(file))
        if not (p == _TMPDIR or p.startswith(_TMPDIR + os.sep)):
            raise PermissionError("filesystem writes are restricted to the sandbox temp dir")
    return _real_open(file, mode, *a, **k)
builtins.open = _guarded_open

SENT = "%SENTINEL%"

def _emit(obj):
    sys.stdout.write(SENT + json.dumps(obj) + "\n")
    sys.stdout.flush()

def main():
    payload = json.loads(sys.stdin.read())
    source = payload["source"]
    xs = payload["xs"]
    ns = {}
    real_stdout = sys.stdout
    sys.stdout = io.StringIO()  # capture/ignore candidate prints
    try:
        exec(compile(source, "<candidate>", "exec"), ns)
        fn = ns.get("transform")
        if not callable(fn):
            sys.stdout = real_stdout
            _emit({"setup_ok": False, "error": "no callable transform(x) defined", "rows": []})
            return
        rows = []
        for x in xs:
            try:
                value = fn(x)
                json.dumps(value)  # ensure serializable
                rows.append({"ok": True, "value": value})
            except (TypeError, ValueError) as e:
                rows.append({"ok": False, "error": "bad/non-serializable result: " + str(e)})
            except Exception as e:
                rows.append({"ok": False, "error": type(e).__name__ + ": " + str(e)})
    finally:
        sys.stdout = real_stdout
    _emit({"setup_ok": True, "error": None, "rows": rows})

try:
    main()
except Exception as e:  # noqa: BLE001
    sys.stdout.write(SENT + json.dumps({"setup_ok": False, "error": type(e).__name__ + ": " + str(e), "rows": []}) + "\n")
finally:
    shutil.rmtree(_TMPDIR, ignore_errors=True)
'''.replace("%SENTINEL%", _SENTINEL)


def _make_limits(timeout_s: float, mem_mb: int):
    def _limits() -> None:  # runs in the child after fork, before exec
        import resource

        def _try(res: int, soft: int, hard: int) -> None:
            # Best-effort: some platforms don't support a given limit (notably
            # RLIMIT_AS on macOS). A failure here must NOT abort the subprocess,
            # so swallow it — the wall-clock timeout (parent-side) and the
            # network/fs guards (in-process) are the platform-independent floor.
            try:
                resource.setrlimit(res, (soft, hard))
            except (ValueError, OSError):
                pass

        cpu = int(timeout_s) + 1
        _try(resource.RLIMIT_CPU, cpu, cpu)
        mem = mem_mb * 1024 * 1024
        _try(resource.RLIMIT_AS, mem, mem)
        fsize = 16 * 1024 * 1024
        _try(resource.RLIMIT_FSIZE, fsize, fsize)

    return _limits


def _kill(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:  # pragma: no cover
        pass


def run_candidate_batch(
    source: str,
    xs: list[dict[str, Any]],
    *,
    timeout_s: float = 5.0,
    mem_mb: int = 512,
) -> list[dict[str, Any]]:
    """Map ``transform`` over many inputs in one subprocess.

    Returns one ``{"ok": bool, "value"|"error"}`` dict per input. If the whole
    subprocess fails (compile error, no transform, timeout, OOM), every input
    gets the same ``ok=False`` error.
    """
    payload = json.dumps({"source": source, "xs": xs})
    cmd = [sys.executable, "-I", "-B", "-c", _DRIVER]
    try:
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, start_new_session=True, preexec_fn=_make_limits(timeout_s, mem_mb),  # noqa: PLW1509
        )
    except OSError as e:  # pragma: no cover
        return [{"ok": False, "error": f"spawn failed: {e}"} for _ in xs]

    try:
        out, err = proc.communicate(payload, timeout=timeout_s + 1.0)
    except subprocess.TimeoutExpired:
        _kill(proc)
        return [{"ok": False, "error": "timeout"} for _ in xs]

    if proc.returncode != 0:
        msg = f"subprocess exited rc={proc.returncode}: {err.strip()[:300]}"
        return [{"ok": False, "error": msg} for _ in xs]

    parsed = None
    for line in reversed(out.splitlines()):
        if line.startswith(_SENTINEL):
            parsed = json.loads(line[len(_SENTINEL):])
            break
    if parsed is None:
        return [{"ok": False, "error": f"no result line; stderr={err[:200]!r}"} for _ in xs]
    if not parsed.get("setup_ok"):
        return [{"ok": False, "error": parsed.get("error", "setup failed")} for _ in xs]
    rows = parsed.get("rows", [])
    if len(rows) != len(xs):  # pragma: no cover - defensive
        return [{"ok": False, "error": "row count mismatch"} for _ in xs]
    return rows


def run_candidate(
    source: str,
    x: dict[str, Any],
    *,
    timeout_s: float = 5.0,
    mem_mb: int = 512,
) -> dict[str, Any]:
    """Run untrusted ``transform(x)`` once; never raises for candidate failures."""
    return run_candidate_batch(source, [x], timeout_s=timeout_s, mem_mb=mem_mb)[0]
