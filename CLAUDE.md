# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A **chaotic attractor explorer**: hand-coded classic systems (Lorenz, Rössler, Chen, …) live alongside a random-search pipeline that discovers new 3D quadratic-polynomial attractors and renders them in vispy. The headline workflow is `python search.py` → `python -m animations.explore`.

## Setup & commands

```bash
uv venv --python 3.11                         # project environment
uv pip install --python .venv/bin/python -e .    # editable install
uv run python search.py --n 1000 --le-min 0.05   # discover new attractors → discovered.jsonl
uv run python -m animations.explore              # cycle through discoveries with ← / →
uv run python -m animations.live_display         # polished fullscreen/gallery display
uv run python -m animations.live_display --windowed  # windowed dev mode
uv run python -m animations.lorenz_3d            # canonical Lorenz, vispy
uv run python animations/lorenz_2d.py            # canonical Lorenz, pygame 2D projection
uv run python animations/random_2d.py            # 2D quadratic-map search (older sibling of search.py)
```

The project is a proper installable package via `pyproject.toml`. Don't add `sys.path.append(os.getcwd())` anywhere — the editable install handles imports. No test suite, no linter config; scripts run directly.

## Architecture

Two packages, one search script.

### `Attractors/` — the load-bearing layer

Every attractor (13 hand-coded + `RandomPolynomial3D`) subclasses `BaseChaoticAttractor` (`Attractors/base_chaotic_attractor.py`). The base class owns:

- `__init__(initial_state=None, **params)` — generic. Subclasses just declare `PARAM_NAMES = ("a", "b", ...)` and `DEFAULT_*` class constants; the base wires defaults automatically. **No `if x is None: x = self.DEFAULT_X` boilerplate in subclasses.**
- `next_state(t, state, parameters)` — the only thing subclasses override, returning `[dx/dt, dy/dt, dz/dt]`.
- `generate_trajectory(t_span, t_steps)` — adaptive `scipy.integrate.solve_ivp`. Use for smooth animation-quality output.
- `integrate_rk4(total_time, dt)` — fixed-step RK4. Use for fast screening of unknown/stiff systems where speed beats precision. Bails out early on divergence.
- `lyapunov_exponent(total_time, dt, renorm_every, d0, ...)` — two-orbit method on top of `_rk4_step`. Returns ~0.84 for Lorenz, ~0.08 for Rössler. Returns `nan` if either orbit overflows `bound`.
- `is_chaotic()` — convenience: bounded + positive LE.
- `generate_perturbed_trajectories(num_trajectories, perturbation_magnitude, ...)` — generates trajectories from perturbed initial states **without permanently mutating** `self.initial_state` (earlier versions of this method had a reset bug; the current implementation restores via `try/finally`).

**Adding a new attractor:** create `Attractors/<name>.py` subclassing `BaseChaoticAttractor`, declare `PARAM_NAMES` + `DEFAULT_*` + `next_state`, and re-export from `Attractors/__init__.py`. ~10 lines total (see `lorenz.py` as the canonical template).

### `Attractors/random_polynomial_3d.py` — the discovery surface

`RandomPolynomial3D(seed=int)` is a `BaseChaoticAttractor` whose dynamics are determined by 30 coefficients sampled from a single seed:

- Each of `dx, dy, dz` is a dot product over the monomial basis `[1, x, y, z, x², y², z², xy, xz, yz]`.
- `SPARSITY = 0.5` zeros out roughly half the coefficients — the well-known 3D chaotic systems are all sparse in this basis, and dense random coefficients almost always diverge. Don't increase sparsity past ~0.6 without re-tuning `SAMPLE_RANGE`; it controls hit rate.
- A single integer **seed reproduces the whole system** — this is why `discovered.jsonl` only needs to store seeds, not the coefficient matrices.

### `search.py` — random-seed search loop

`evaluate(seed)` does prefilter → Lyapunov in this order:

1. `integrate_rk4` for `total_time` seconds. Reject if it diverges before reaching the full horizon, exceeds `bound`, or collapses to a fixed point (tiny range across the last 25% of the trajectory).
2. `lyapunov_exponent` (also RK4-based). Reject if LE < `le_min` or non-finite.

Survivors are appended as JSON to `discovered.jsonl` with their `seed`, `lyapunov`, and `bbox`. Typical hit rate at default settings: ~1-2% — expect tens of survivors per thousand seeds.

### `animations/` — visualizers

All three vispy/pygame scripts consume `Attractors/` rather than re-deriving dynamics inline:

- `lorenz_3d.py` — vispy turntable with multi-particle Euler integration of `LorenzAttractor.next_state`. Trails are rendered as fading line segments.
- `lorenz_2d.py` — pygame 2D projection, z used for brightness + line-thickness depth cues, also driven by `LorenzAttractor.next_state`.
- `explore.py` — **the seed-browsing UI**. Reads `discovered.jsonl`, instantiates each survivor as a `RandomPolynomial3D(seed=...)`, animates it with scatter + deque-trail visuals. Keys: `← / →` navigate, `space` resets the particle cloud, `q / Esc` quit.
- `live_display.py` — **the polished fullscreen/gallery UI**. Sorts discoveries by Lyapunov exponent, runs vectorized RK4 integration for smoother motion, renders glowing particle heads plus velocity-reactive gradient trail segments, rotates the camera slowly, cycles attractors automatically, and supports curated palettes. Keys: `f` fullscreen, `← / →` navigate, `space` pause, `r` reset, `p` palette, `+ / -` velocity, `h` HUD, `q / Esc` quit.
- `random_2d.py` — older standalone script that does the same random-search idea but for 2D *discrete maps* (not ODEs). Self-contained; kept for the 2D aesthetic it produces. Not integrated with `BaseChaoticAttractor` because the discrete-map case doesn't fit that interface.

## Conventions

- **Subclasses are kwargs-only.** `LorenzAttractor(sigma=12)` works; `LorenzAttractor(initial_state, 12, 28, 2.67)` no longer does. The generic base `__init__` accepts `**params` keyed by `PARAM_NAMES`.
- **Parameters are stored as a positional tuple** in the order of `PARAM_NAMES` and passed through `solve_ivp` as `args=(self.parameters,)`. `next_state` unpacks positionally — keep that contract intact.
- **All attractors are 3D.** State is `[x, y, z]`. The polynomial basis in `random_polynomial_3d.py` and the `next_state` shape are hardcoded for 3 dimensions; lifting to nD would require generalizing both.
- **Reproducibility.** A discovered attractor is just its seed. Don't add stateful randomness to `RandomPolynomial3D.__init__` paths beyond what the seed controls.
