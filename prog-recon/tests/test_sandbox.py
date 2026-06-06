"""Candidate sandbox tests (CHECKPOINT 2).

These confirm the locked-down subprocess: it returns values for well-behaved
candidates, and contains misbehaving ones (timeout, network, fs writes, OOM,
bad output) without raising.
"""

from progrecon.harness import sandbox


def test_normal_candidate():
    src = "def transform(x):\n    return {'out_0': x['a'] + 1}"
    r = sandbox.run_candidate(src, {"a": 41}, timeout_s=5)
    assert r == {"ok": True, "value": {"out_0": 42}}


def test_infinite_loop_is_killed():
    src = "def transform(x):\n    while True:\n        pass"
    r = sandbox.run_candidate(src, {"a": 1}, timeout_s=2)
    assert r["ok"] is False
    assert "timeout" in r["error"] or "rc=" in r["error"]


def test_network_is_blocked():
    src = (
        "import socket\n"
        "def transform(x):\n"
        "    s = socket.socket()\n"
        "    s.connect(('1.1.1.1', 80))\n"
        "    return {}\n"
    )
    r = sandbox.run_candidate(src, {"a": 1}, timeout_s=5)
    assert r["ok"] is False
    assert "network" in r["error"].lower()


def test_fs_write_outside_temp_blocked():
    src = (
        "def transform(x):\n"
        "    open('/tmp/progrecon_should_not_exist.txt', 'w').write('x' * 50)\n"
        "    return {}\n"
    )
    r = sandbox.run_candidate(src, {"a": 1}, timeout_s=5)
    assert r["ok"] is False
    assert "writes are restricted" in r["error"] or "Permission" in r["error"]


def test_no_transform_defined():
    r = sandbox.run_candidate("y = 5", {"a": 1}, timeout_s=5)
    assert r["ok"] is False
    assert "transform" in r["error"]


def test_candidate_exception_is_reported():
    src = "def transform(x):\n    return {'out_0': 1 / 0}"
    r = sandbox.run_candidate(src, {"a": 1}, timeout_s=5)
    assert r["ok"] is False
    assert "ZeroDivision" in r["error"]


def test_non_serializable_result_reported():
    src = "def transform(x):\n    return {'out_0': set([1, 2, 3])}"
    r = sandbox.run_candidate(src, {"a": 1}, timeout_s=5)
    assert r["ok"] is False


def test_candidate_stdout_is_isolated():
    src = "def transform(x):\n    print('noise that must not corrupt the result')\n    return {'out_0': 7}"
    r = sandbox.run_candidate(src, {"a": 1}, timeout_s=5)
    assert r == {"ok": True, "value": {"out_0": 7}}


def test_memory_cap_contains_allocation():
    # Try to allocate ~400MB with a 64MB cap; expect a contained failure.
    src = "def transform(x):\n    big = bytearray(400 * 1024 * 1024)\n    return {'n': len(big)}"
    r = sandbox.run_candidate(src, {"a": 1}, timeout_s=5, mem_mb=64)
    assert r["ok"] is False
