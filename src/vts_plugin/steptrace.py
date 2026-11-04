"""Opt-in engine step trace (used to split step time into scheduler / GPU / gc).

VTS_STEP_TRACE=<dir> wraps EngineCore.step / step_with_batch_queue, Scheduler.schedule and
Scheduler.update_from_output in the engine-core process, and registers a gc callback. One JSON
line per engine step goes to <dir>/steps_<pid>.jsonl:
  {"t": wall time at the end of the step, "wall": step ms, "sched": schedule() ms,
   "update": update_from_output() ms, "gc": garbage-collection ms during the step,
   "waiting": len(waiting) and "running": len(running) when schedule() was called,
   "tokens": total scheduled tokens, "new": newly scheduled requests}
No GPU synchronisation is added. Unset = nothing is patched.
"""
import gc
import json
import os
import time

_STATE = {"sched": 0.0, "update": 0.0, "gc": 0.0, "gc_t0": None, "waiting": 0, "running": 0,
          "tokens": 0, "new": 0}


def _gc_cb(phase, info):
    if phase == "start":
        _STATE["gc_t0"] = time.perf_counter()
    elif _STATE["gc_t0"] is not None:
        _STATE["gc"] += 1000 * (time.perf_counter() - _STATE["gc_t0"])
        _STATE["gc_t0"] = None


def _wrap_schedule(orig):
    def schedule(self, *args, **kwargs):
        _STATE["waiting"] = len(self.waiting)
        _STATE["running"] = len(self.running)
        t0 = time.perf_counter()
        out = orig(self, *args, **kwargs)
        _STATE["sched"] += 1000 * (time.perf_counter() - t0)
        _STATE["tokens"] += int(getattr(out, "total_num_scheduled_tokens", 0) or 0)
        _STATE["new"] += len(getattr(out, "scheduled_new_reqs", ()) or ())
        return out
    return schedule


def _wrap_update(orig):
    def update_from_output(self, *args, **kwargs):
        t0 = time.perf_counter()
        out = orig(self, *args, **kwargs)
        _STATE["update"] += 1000 * (time.perf_counter() - t0)
        return out
    return update_from_output


def _wrap_step(orig, path):
    def step(self, *args, **kwargs):
        for k in ("sched", "update", "gc", "tokens", "new"):
            _STATE[k] = 0
        t0 = time.perf_counter()
        out = orig(self, *args, **kwargs)
        wall = 1000 * (time.perf_counter() - t0)
        with open(path, "a") as f:
            f.write(json.dumps({"t": round(time.time(), 4), "wall": round(wall, 3),
                                "sched": round(_STATE["sched"], 3), "update": round(_STATE["update"], 3),
                                "gc": round(_STATE["gc"], 3), "waiting": _STATE["waiting"],
                                "running": _STATE["running"], "tokens": _STATE["tokens"],
                                "new": _STATE["new"]}) + "\n")
        return out
    return step


def install(engine_cls, scheduler_cls, out_dir):
    """Patch the classes once; returns True if it patched."""
    if getattr(engine_cls, "_vts_step_traced", False):
        return False
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"steps_{os.getpid()}.jsonl")
    scheduler_cls.schedule = _wrap_schedule(scheduler_cls.schedule)
    scheduler_cls.update_from_output = _wrap_update(scheduler_cls.update_from_output)
    for name in ("step", "step_with_batch_queue"):
        if hasattr(engine_cls, name):
            setattr(engine_cls, name, _wrap_step(getattr(engine_cls, name), path))
    if _gc_cb not in gc.callbacks:
        gc.callbacks.append(_gc_cb)
    engine_cls._vts_step_traced = True
    return True


def maybe_install():
    out_dir = os.environ.get("VTS_STEP_TRACE")
    if not out_dir:
        return False
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.engine.core import EngineCore

    return install(EngineCore, Scheduler, out_dir)
