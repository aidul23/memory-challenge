"""The multi-seed sweep: workloads are generated once and reloaded, every
system runs on every seed, and repeats are logged and scored for noise."""

import json
import sys

from runner import sweep
from systems.ledger.system import Ledger


def test_sweep_generates_runs_and_measures_repeats(tmp_path, monkeypatch, capsys):
    # A deterministic stand-in for an LLM system: repeats must agree exactly.
    monkeypatch.setitem(sweep.UNCACHED, "ledger", Ledger)
    monkeypatch.setattr(sys, "argv", ["sweep", "--seeds", "0", "1", "--systems", "no_memory", "ledger",
                                      "--repeats", "2", "--workloads", str(tmp_path / "w"),
                                      "--out", str(tmp_path / "runs")])
    sweep.main()

    assert sorted(p.name for p in (tmp_path / "w").iterdir()) == ["synthetic-small-s0.json",
                                                                   "synthetic-small-s1.json"]
    logs = sorted(p.name for p in (tmp_path / "runs" / "synthetic-small-s0").iterdir())
    assert logs == ["ledger.json", "ledger__r2.json", "no_memory.json"]  # deterministic systems run once

    summary = json.loads((tmp_path / "runs" / "sweep-small.json").read_text())
    assert summary["systems"]["no_memory"]["accuracy_pct"] == [0.0, 0.0]
    assert summary["systems"]["ledger"]["noise"]["flipped"] == 0
    assert "run-to-run noise" in capsys.readouterr().out

    # A second sweep reloads the saved workloads rather than regenerating them.
    before = (tmp_path / "w" / "synthetic-small-s0.json").stat().st_mtime_ns
    monkeypatch.setattr(sys, "argv", ["sweep", "--seeds", "0", "--systems", "no_memory",
                                      "--workloads", str(tmp_path / "w"), "--out", str(tmp_path / "runs")])
    sweep.main()
    assert (tmp_path / "w" / "synthetic-small-s0.json").stat().st_mtime_ns == before
