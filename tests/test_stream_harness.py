"""The shipped streaming path must be drivable and auditable from one command.

These tests are deliberately offline: they exercise the parser and the gate's
decision logic on captured driver output, so they need no GPU, no model and no
driver binary. What they protect is that the numbers `sabah run`/`sabah gate`
print keep meaning what the README says they mean.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sabah.tools import stream  # noqa: E402

DRIVER_OUT = """\
sabah-diag: prompt step 0 argmax 11 fed 11  11000.0 ms
sabah-diag: prompt step 1 argmax 42 fed 42  200.0 ms
sabah-diag: prompt step 2 argmax 43 fed 43  210.0 ms
sabah-diag: prompt step 3 argmax 44 fed 44  230.0 ms
SABAH_DIAG calls=432 tokens=432 lookups=3456 overflow=0 verify_ok=0 verify_fail=0
SABAH_DISK unbuffered_reads=300 bytes=192000000 waves=150 mapped_fallbacks=0 \
staging_bytes=33554432 direct_io=1 depth=8 lanes=8 lane_reads=300 sync=0 pending=300
SABAH_IO read_ms=90.0 wave_ms=120.0 overlap=3.84 reads_per_wave=2.00
SABAH_PHASE calls=432 ids=84.0ms fetch=363.0ms wait=1.5ms slot=28.5ms ptr=4.5ms \
prime=4.2ms | per_call us: ids=194 fetch=840 wait=3 slot=66 ptr=10 prime=10 | \
accounting=485.7ms
SABAH_IDS reused=288 group_max=3 check=1 check_ok=288 check_fail=0
"""


def test_parse_driver_separates_prefill_from_decode():
    r = stream.parse_driver(DRIVER_OUT)
    # step 0 is the prefill: it must not be counted as a decoded token, and its
    # 11 s must not enter the per-token figure.
    assert r["n_decodes"] == 3
    assert abs(r["decode_ms_sum"] - 640.0) < 1e-6
    assert abs(r["decode_ms_median"] - 210.0) < 1e-6
    assert abs(r["decode_ms_per_token"] - 213.333333) < 1e-3
    assert abs(r["decode_tok_per_s"] - 4.6875) < 1e-3


def test_parse_driver_reads_the_runtime_counters():
    r = stream.parse_driver(DRIVER_OUT)
    assert r["calls"] == 432
    assert r["tokens"] == 432
    assert r["lookups"] == 3456
    assert r["read_bytes"] == 192000000
    assert r["unbuffered_reads"] == 300
    assert r["waves"] == 150
    assert r["mapped_fallbacks"] == 0
    assert r["verify_ok"] == 0 and r["verify_fail"] == 0
    assert r["phase_ids_ms"] == 84.0
    assert r["phase_fetch_ms"] == 363.0
    assert r["phase_prime_ms"] == 4.2
    assert r["id_check"] == 1
    assert r["id_check_ok"] == 288 and r["id_check_fail"] == 0
    assert abs(r["io_overlap"] - 3.84) < 1e-6


def test_parse_driver_on_empty_output_is_all_none():
    r = stream.parse_driver("")
    assert r["n_decodes"] == 0
    assert "decode_ms_per_token" not in r


def test_missing_status_file_is_not_an_error():
    assert stream.status("Z:/definitely/not/here.json") == {}
    assert stream.status("") == {}


def test_gate_fails_when_the_runtime_serves_a_wrong_byte():
    # A verify_fail is the whole point of the gate: it must not pass.
    good = {"calls": 100, "verify_ok": 12, "verify_fail": 0, "id_check_ok": 5,
            "id_check_fail": 0, "mapped_fallbacks": 0, "exit_code": 0}
    assert all(ok for _, ok, _ in stream.gate_checks(good))
    bad = dict(good, verify_fail=1)
    assert [label for label, ok, _ in stream.gate_checks(bad) if not ok] == \
        ["every served byte matched the GGUF"]


def test_gate_does_not_fail_on_documented_over_budget_allocations():
    # `overflow` counts over-budget allocations, not bad expert ids: this is the
    # shape of a real passing gate run on the reference rig (448 over-budget
    # allocations alongside 2000 byte-exact verifications), so it must pass.
    rec = {"calls": 1152, "verify_ok": 2000, "verify_fail": 0, "id_check_ok": 768,
           "id_check_fail": 0, "mapped_fallbacks": 0, "exit_code": 0,
           "overflow": 448}
    assert all(ok for _, ok, _ in stream.gate_checks(rec))
    assert not any("escaped" in label for label, _, _ in stream.gate_checks(rec))
    assert ("over-budget allocations", 448) in stream.gate_notes(rec)


def test_matrix_labels_are_unique():
    labels = [row[0] for row in stream.MATRIX]
    assert len(labels) == len(set(labels))
    for label, env, _flags, tokens in stream.MATRIX:
        assert isinstance(label, str) and label
        assert isinstance(env, dict)
        assert tokens is None or (isinstance(tokens, int) and tokens > 0)
    # one row must be long enough to reach the steady state, or the matrix is
    # reporting the warm-up as if it were the decode rate
    assert any((t or 0) >= 256 for _, _, _, t in stream.MATRIX)


def test_regimes_split_fill_warmup_and_steady_state(tmp_path=None):
    import struct
    import tempfile
    d = tempfile.mkdtemp()
    # step 0 = the fill; then a slow warm-up and a flat steady state
    ms = [11200.0] + [200.0] * 128 + [67.0] * 128
    with open(os.path.join(d, "step_ms.f64"), "wb") as f:
        f.write(struct.pack("<%dd" % len(ms), *ms))
    r = stream.regimes(d)
    assert r["fill_ms"] == 11200.0
    assert r["decodes"] == 256
    assert r["warmup"]["n"] == 128 and r["warmup"]["median_ms"] == 200.0
    assert r["steady"]["n"] == 128 and r["steady"]["median_ms"] == 67.0
    assert abs(r["steady"]["tok_per_s"] - 14.925) < 1e-2
    # 128*200 + 128*67 = 34176 ms of decode, plus 11.2 s of fill: 45376 ms
    assert abs(r["decode_only_tok_per_s"] - 7.491) < 1e-2
    assert abs(r["end_to_end_tok_per_s"] - 5.642) < 1e-2


def test_regimes_on_a_short_run_has_no_steady_state(tmp_path=None):
    import struct
    import tempfile
    d = tempfile.mkdtemp()
    ms = [11000.0] + [200.0] * 4
    with open(os.path.join(d, "step_ms.f64"), "wb") as f:
        f.write(struct.pack("<%dd" % len(ms), *ms))
    r = stream.regimes(d)
    assert "steady" not in r
    assert r["warmup"]["median_ms"] == 200.0


def test_missing_step_file_is_not_an_error():
    assert stream.step_ms("Z:/definitely/not/here") == []
    assert stream.regimes("Z:/definitely/not/here") == {}
