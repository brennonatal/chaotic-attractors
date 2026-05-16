"""Dump discovered attractor coefficients to JSON for the three.js port.

`numpy.random.default_rng` uses PCG64, which is not worth reproducing in JS.
Instead we materialize every survivor's 3x10 coefficient matrix here and
ship it as static data the web client just reads.

Usage:
    uv run python tools/dump_coeffs.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from Attractors import RandomPolynomial3D


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default="discovered.jsonl",
                        help="JSONL produced by search.py")
    parser.add_argument("--dst", default="web/public/attractors.json",
                        help="output JSON for the web client")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    out: list[dict] = []
    with src.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            attractor = RandomPolynomial3D(seed=entry["seed"])
            out.append({
                "seed": entry["seed"],
                "lyapunov": entry["lyapunov"],
                "bbox": entry["bbox"],
                "initialState": attractor.initial_state.tolist(),
                "coeffs": attractor.parameters[0].tolist(),
            })

    out.sort(key=lambda e: float(e.get("lyapunov", 0.0)), reverse=True)

    with dst.open("w") as f:
        json.dump(out, f)

    print(f"Wrote {len(out)} attractors → {dst}")


if __name__ == "__main__":
    main()
