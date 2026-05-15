"""Search for chaotic 3D polynomial attractors by random seed.

Run as: ``python search.py --n 500`` to try 500 random seeds. Survivors land
in ``discovered.jsonl`` (one JSON object per line) and can be visualized with
``python -m animations.explore``.
"""

import argparse
import json
import time

import numpy as np

from Attractors import RandomPolynomial3D


def evaluate(seed, total_time=30.0, bound=1e3, le_min=0.05):
    """Try to accept a seed. Returns a dict on accept, ``None`` on reject."""
    attractor = RandomPolynomial3D(seed=seed)

    # Fast RK4 prefilter — bails out instantly on divergent / stiff systems
    # that would make solve_ivp grind for seconds.
    traj = attractor.integrate_rk4(total_time=total_time, dt=0.01, bound=bound)
    if traj.shape[1] < int(0.9 * total_time / 0.01):
        return None  # diverged before reaching full horizon
    if not np.all(np.isfinite(traj)) or np.abs(traj).max() > bound:
        return None

    # Reject if it collapses to a fixed point (tiny range across the tail).
    tail = traj[:, int(0.75 * traj.shape[1]):]
    tail_range = tail.max(axis=1) - tail.min(axis=1)
    if tail_range.max() < 0.05:
        return None

    le = attractor.lyapunov_exponent(total_time=total_time)
    if not np.isfinite(le) or le < le_min:
        return None

    bbox = [[float(traj[d].min()), float(traj[d].max())] for d in range(3)]
    return {"seed": int(seed), "lyapunov": float(le), "bbox": bbox}


def search(n, le_min, out_path, total_time=40.0, bound=1e3):
    rng = np.random.default_rng()
    accepted = 0
    started = time.time()
    with open(out_path, "w") as f:
        for i in range(n):
            seed = int(rng.integers(0, 2**31 - 1))
            result = evaluate(seed, total_time=total_time, bound=bound, le_min=le_min)
            if result is None:
                continue
            f.write(json.dumps(result) + "\n")
            f.flush()
            accepted += 1
            print(
                f"[{i + 1}/{n}] accepted seed={result['seed']} "
                f"λ={result['lyapunov']:.3f}  ({accepted} total, "
                f"{time.time() - started:.1f}s elapsed)"
            )
    print(f"\nDone: {accepted}/{n} accepted in {time.time() - started:.1f}s "
          f"-> {out_path}")
    return accepted


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=200, help="number of seeds to try")
    p.add_argument("--le-min", type=float, default=0.05,
                   help="minimum Lyapunov exponent for acceptance")
    p.add_argument("--out", type=str, default="discovered.jsonl",
                   help="output path (JSONL)")
    p.add_argument("--total-time", type=float, default=40.0,
                   help="integration horizon per candidate")
    p.add_argument("--bound", type=float, default=1e3,
                   help="reject if any coordinate exceeds this magnitude")
    args = p.parse_args()
    search(args.n, args.le_min, args.out, total_time=args.total_time, bound=args.bound)
