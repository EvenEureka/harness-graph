import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("make_c1_pool", Path(__file__).resolve().parents[1] / "scripts/make_c1_pool.py")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)

GOLD = {"apply_ok": True, "timeout": False, "resolved": True, "f2p_pass": 2, "f2p_total": 2, "p2p_pass": 5, "p2p_total": 5}
EMPTY = {"bug_apply_fail": False, "timeout": False, "resolved": False, "f2p_pass": 0, "f2p_missing": 0, "f2p_total": 2}


def grader(gold=None, empty=None):
    return lambda t, sif, mode: {**GOLD, **(gold or {})} if mode == "gold" else {**EMPTY, **(empty or {})}


def test_a_repository_is_matched_by_its_name_at_any_revision():
    assert P.names("pallets__click.fde47b4b") == {"pallets__click", "click"}
    assert P.names("commit-0/click") == {"click"}
    assert P.names("swesmith/pdfminer__pdfminer.six.1a8bd2f7") == {"pdfminer__pdfminer.six", "pdfminer.six"}
    assert P.strategy("a__b.1234.combine_module__x1") == "combine_module"


def test_controls_need_a_resolving_fix_and_an_observed_failure():
    assert P.control({}, "sif", grader())[0]
    cases = {"gold does not resolve": grader(gold={"resolved": False}), "gold run timed out": grader(gold={"timeout": True}),
             "gold patch does not apply": grader(gold={"apply_ok": False}), "defect does not apply": grader(empty={"bug_apply_fail": True}),
             "unchanged run timed out": grader(empty={"timeout": True}), "unchanged repository resolves": grader(empty={"resolved": True}),
             "no failing test observed to fail": grader(empty={"f2p_missing": 2})}
    for why, g in cases.items():
        assert P.control({}, "sif", g)[:2] == (False, why)


def test_an_exception_in_a_control_is_tried_twice_then_fails():
    calls = []

    def flaky(t, sif, mode):
        calls.append(mode)
        if len(calls) == 1:
            raise RuntimeError("apptainer")
        return GOLD if mode == "gold" else EMPTY
    assert P.control({}, "sif", flaky)[0] and len(calls) == 3

    def broken(t, sif, mode):
        raise RuntimeError("apptainer")
    ok, why, _ = P.control({}, "sif", broken)
    assert not ok and why.startswith("exception")


def test_the_probe_log_allows_a_second_attempt_only_after_an_infrastructure_failure(tmp_path):
    f = tmp_path / "log.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in [
        {"repo_key": "a", "attempt": 1, "infrastructure": True}, {"repo_key": "a", "attempt": 2, "accepted": True},
        {"repo_key": "b", "attempt": 1, "accepted": False}]))
    assert P.read_log(f)["a"]["attempt"] == 2
    f.write_text("\n".join(json.dumps(r) for r in [{"repo_key": "b", "attempt": 1}, {"repo_key": "b", "attempt": 2}]))
    with pytest.raises(AssertionError):
        P.read_log(f)
