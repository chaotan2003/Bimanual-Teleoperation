"""Measure where the JAKA K1 CasADi cycle actually spends time, on real XR hardware.

Why this exists: an offline benchmark (stub XR client, see the numbers below) puts the
whole cycle at ~2.8 ms inside a 33 ms budget, so IPOPT is NOT the bottleneck. The one
part no offline harness can measure is xrobotoolkit_sdk -- 9 pybind11 calls per frame
into the PICO PC Service, any of which may block until the device publishes a new sample.

Offline reference (bimanual-casadi env, OMP_NUM_THREADS=1, stub XR):
    cycle with meshcat, hand at 1200 mm/s   p50 2.77  p95 3.13  max 3.68 ms
      of which IPOPT (8 iterations)                1.90 ms
      of which viz.display                         0.62 ms
      of which everything else                     0.28 ms
    -> 30 Hz budget 33.33 ms is only 9% used; raising --frequency is nearly free.

Run on the teleop machine with the headset connected:
    python scripts/simulation/probe_jaka_latency.py --seconds 20
    python scripts/simulation/probe_jaka_latency.py --seconds 20 --frequency 90
Compare the two: if the frame interval tracks --frequency, the loop was never the limit;
if it pins to ~16.6 or ~33 ms regardless, an SDK read is blocking on the device rate.
"""

import os

# Single-thread the BLAS before numpy: see casadi_arm_ik.py docstring.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import gc
import sys
import time
import webbrowser
from collections import defaultdict

# Runnable as a script (python scripts/simulation/probe_jaka_latency.py), where sys.path[0]
# is this directory rather than the repo root that 'scripts.simulation...' needs.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import tyro

from scripts.simulation.teleop_jaka_k1_casadi import (
    JAKA_K1_MANIPULATOR_CONFIG,
    JAKA_K1_Q_INIT,
    JAKA_K1_URDF_PATH,
)
from xrobotoolkit_teleop.simulation.casadi_teleop_controller import CasadiTeleopController

# Every XrClient method the JAKA SWGR path touches, plus the controller stages around them.
XR_METHODS = (
    "get_body_tracking_data",
    "get_key_value_by_name",
    "get_pose_by_name",
    "get_motion_tracker_data",
)
STAGES = ("_update_ik", "_solve_ik", "_send_command", "_update_viz", "_read_body_upper_limbs")


class Recorder:
    """Per-name latency samples, collected by wrapping callables in place."""

    def __init__(self):
        self.t = defaultdict(list)

    def wrap(self, obj, name):
        """Shadow ``obj.name`` with a timing wrapper (instance attr wins over class method)."""
        try:
            fn = getattr(obj, name)
        except AttributeError:
            return False
        rec = self.t

        def timed(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                rec[name].append((time.perf_counter() - t0) * 1e3)

        setattr(obj, name, timed)
        return True


def _pct(a, p):
    a = np.sort(a)
    return a[min(len(a) - 1, int(len(a) * p))]


def _row(label, samples, frames):
    a = np.asarray(samples)
    print(
        f"  {label:<30}{len(a) / frames:6.1f}  {a.mean():8.3f}{_pct(a, .50):9.3f}"
        f"{_pct(a, .95):9.3f}{a.max():9.3f}{a.sum() / frames:9.3f}"
    )


def main(seconds: float = 20.0, frequency: float = 30.0, no_viz: bool = False):
    webbrowser.open = lambda *a, **k: None  # headless probe: do not spawn a browser
    if no_viz:
        CasadiTeleopController._init_viz = lambda self: None
    ctrl = CasadiTeleopController(
        robot_urdf_path=JAKA_K1_URDF_PATH,
        manipulator_config=JAKA_K1_MANIPULATOR_CONFIG,
        q_init=JAKA_K1_Q_INIT,
        left_frame="lt",
        right_frame="rt",
        frequency=frequency,
    )
    if no_viz:
        ctrl._update_viz = lambda: None

    rec = Recorder()
    for name in XR_METHODS:
        if not rec.wrap(ctrl.xr_client, name):
            print(f"  (XrClient has no {name}; skipped)")
    for name in STAGES:
        rec.wrap(ctrl, name)

    # Settle the warm-start chain before measuring, so iteration counts are steady-state.
    ctrl.xr_client.get_key_value_by_name("left_grip")
    for _ in range(30):
        ctrl._update_ik()
    rec.t.clear()

    print(f"\nMoving the controllers for {seconds:.0f} s at {frequency:.0f} Hz -- record now.\n")
    intervals, gc_before = [], gc.get_count()
    deadline = time.perf_counter()
    t_end = time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        deadline += 1.0 / frequency
        t0 = time.perf_counter()
        ctrl._update_ik()
        ctrl._send_command()
        intervals.append((time.perf_counter() - t0) * 1e3)
        lag = deadline - time.perf_counter()
        if lag > 0:
            time.sleep(lag)
        else:
            deadline = time.perf_counter()  # fell behind: resync, do not try to catch up

    frames = len(intervals)
    budget = 1000.0 / frequency
    iv = np.asarray(intervals)
    over = int((iv > budget).sum())
    print("=" * 100)
    print(f"FRAME INTERVAL  ({frames} frames in {seconds:.0f} s, budget {budget:.2f} ms @ {frequency:.0f} Hz)")
    print(
        f"  compute-only mean={iv.mean():7.3f}  p50={_pct(iv, .50):7.3f}  "
        f"p95={_pct(iv, .95):7.3f}  p99={_pct(iv, .99):7.3f}  max={iv.max():7.3f} ms"
    )
    print(f"  frames over budget: {over} ({100.0 * over / frames:.1f}%)   effective rate: {frames / seconds:.2f} Hz")
    print(f"  gc counts before/after: {gc_before} -> {gc.get_count()}")

    print("=" * 100)
    print(f"  {'stage':<30}{'calls/fr':>8}{'mean':>9}{'p50':>9}{'p95':>9}{'max':>9}{'ms/frame':>10}")
    print("  " + "-" * 94)
    for name in STAGES:
        if rec.t.get(name):
            _row(name, rec.t[name], frames)
    print("  " + "-" * 94)
    sdk_total = []
    for name in XR_METHODS:
        if rec.t.get(name):
            _row(f"  sdk:{name}", rec.t[name], frames)
            sdk_total.append(np.asarray(rec.t[name]).sum())
    print("  " + "-" * 94)
    if sdk_total:
        print(f"  {'ALL SDK CALLS':<30}{'':>8}{'':>9}{'':>9}{'':>9}{'':>9}{sum(sdk_total) / frames:9.3f}")
    if ctrl.use_swgr and not rec.t.get("get_pose_by_name"):
        # Without the grip held, _update_ik skips _update_swgr_target entirely, so the
        # most expensive per-frame reads never happen and the table looks artificially free.
        print(
            "\n  !! grip was never engaged during recording: the SWGR retarget path"
            "\n     (get_pose_by_name x2/frame) never ran, so this table UNDER-reports."
            "\n     Hold both grips and move your hands while recording."
        )
    print("=" * 100)
    print("Reading it:")
    print("  * 'sdk:* ms/frame' dominates  -> the PICO bridge is the limit; batch reads off-thread")
    print("  * '_solve_ik' dominates       -> IPOPT; check iters and the BLAS env vars")
    print("  * '_update_viz' dominates      -> meshcat; run with --no-viz to confirm")
    print("  * compute p50 is small but max spikes -> GC or the OS scheduler; try gc.freeze()")
    ctrl._stop_event.set()


if __name__ == "__main__":
    tyro.cli(main)
