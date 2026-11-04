"""Engine step trace (steptrace.py): one JSON line per step with the scheduler's queue sizes,
scheduled tokens and the time spent in schedule() / update_from_output(); results unchanged;
idempotent; nothing patched when the env var is unset."""
import json
import os
import time
import types

import pytest

from vts_plugin import steptrace


class FakeScheduler:
    def __init__(self):
        self.waiting, self.running = [1, 2, 3], [4]

    def schedule(self):
        time.sleep(0.002)
        return types.SimpleNamespace(total_num_scheduled_tokens=2048, scheduled_new_reqs=[1, 2])

    def update_from_output(self, out):
        return {"ok": out}


class FakeEngine:
    def __init__(self):
        self.scheduler = FakeScheduler()

    def step(self):
        out = self.scheduler.schedule()
        return self.scheduler.update_from_output(out.total_num_scheduled_tokens), True


def test_step_trace_records_each_step(tmp_path):
    eng_cls = type("Eng1", (FakeEngine,), {})
    sch_cls = type("Sch1", (FakeScheduler,), {})
    assert steptrace.install(eng_cls, sch_cls, str(tmp_path))
    assert not steptrace.install(eng_cls, sch_cls, str(tmp_path))
    e = eng_cls()
    e.scheduler = sch_cls()
    assert e.step() == ({"ok": 2048}, True)
    assert e.step() == ({"ok": 2048}, True)
    (f,) = [p for p in os.listdir(tmp_path) if p.startswith("steps_")]
    rows = [json.loads(line) for line in open(tmp_path / f)]
    assert len(rows) == 2
    for r in rows:
        assert (r["waiting"], r["running"], r["tokens"], r["new"]) == (3, 1, 2048, 2)
        assert r["sched"] >= 1.5 and r["wall"] >= r["sched"] and r["update"] >= 0 and r["gc"] >= 0


def test_unset_env_patches_nothing(monkeypatch):
    monkeypatch.delenv("VTS_STEP_TRACE", raising=False)
    assert steptrace.maybe_install() is False
