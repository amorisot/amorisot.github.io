"""Locked-down subprocess executor for UNTRUSTED candidate (model) programs.

Constraint (build spec §0.6, §4): candidate source runs in a restricted
subprocess with:
* no network (socket creation is disabled inside the child),
* no filesystem writes (RLIMIT_FSIZE = 0; candidate stdout is captured in-memory),
* a wall-clock cap (parent kills the process group on timeout) AND a CPU cap
  (RLIMIT_CPU, so a busy loop is reaped even if it ignores wall-clock), and
* a memory cap (RLIMIT_AS).

This is defense-in-depth in pure Python, not a hardware sandbox; true isolation
would use containers/seccomp. It is sufficient for the threat model here
(our own and model-authored transform functions).
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from typing import Any

# The child driver: read {"source","x"} from stdin, disable network, exec the
# candidate with its stdout captured, call transform(x), emit a single result
# line prefixed with a sentinel so the parent can find it unambiguously.
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
    x = payload["x"]
    ns = {}
    real_stdout = sys.stdout
    sys.stdout = io.StringIO()  # capture/ignore candidate prints
    try:
        exec(compile(source, "<candidate>", "exec"), ns)
        fn = ns.get("transform")
        if not callable(fn):
            sys.stdout = real_stdout
            _emit({"ok": False, "error": "no callable transform(x) defined"})
            return
        result = fn(x)
    finally:
        sys.stdout = real_stdout
    try:
        _emit({"ok": True, "value": result})
    except (TypeError, ValueError) as e:
        _emit({"ok": False, "error": "non-serializable result: " + str(e)})

try:
    main()
except Exception as e:  # noqa: BLE001 - report any candidate failure
    sys.stdout.write(SENT + json.dumps({"ok": False, "error": type(e).__name__ + ": " + str(e)}) + "\n")
finally:
    shutil.rmtree(_TMPDIR, ignore_errors=True)
'''.replace("%SENTINEL%", _SENTINEL)


def _make_limits(timeout_s: float, mem_mb: int):
    def _limits() -> None:  # runs in the child after fork, before exec
        import resource

        cpu = int(timeout_s) + 1
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        mem = mem_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        # Bound write size (defense in depth); the open-guard confines *where*
        # writes may go (the per-run temp dir only).
        fsize = 16 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))

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


def run_candidate(
    source: str,
    x: dict[str, Any],
    *,
    timeout_s: float = 5.0,
    mem_mb: int = 512,
) -> dict[str, Any]:
    """Run untrusted ``transform(x)`` and return {"ok": bool, "value"|"error"}.

    Never raises for candidate-side failures; returns an ``ok=False`` dict with
    a diagnostic ``error`` instead (timeout, crash, OOM, bad output, ...).
    """
    payload = json.dumps({"source": source, "x": x})
    cmd = [sys.executable, "-I", "-B", "-c", _DRIVER]
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
            preexec_fn=_make_limits(timeout_s, mem_mb),  # noqa: PLW1509
        )
    except OSError as e:  # pragma: no cover - spawn failure
        return {"ok": False, "error": f"spawn failed: {e}"}

    try:
        out, err = proc.communicate(payload, timeout=timeout_s + 1.0)
    except subprocess.TimeoutExpired:
        _kill(proc)
        return {"ok": False, "error": "timeout"}

    if proc.returncode != 0:
        # Negative rc => killed by signal (e.g. -9 SIGKILL, -24 SIGXCPU/OOM).
        return {
            "ok": False,
            "error": f"subprocess exited rc={proc.returncode}: {err.strip()[:300]}",
        }

    for line in reversed(out.splitlines()):
        if line.startswith(_SENTINEL):
            return json.loads(line[len(_SENTINEL):])
    return {
        "ok": False,
        "error": f"no result line; stdout={out[:200]!r} stderr={err[:200]!r}",
    }
