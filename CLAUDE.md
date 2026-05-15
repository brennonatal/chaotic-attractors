# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup & commands

```bash
pip install -r requirements.txt   # numpy, scipy, pygame, PyQt5, vispy
```

`requirements.txt` does **not** list `tensorflow` or `scikit-learn`, but the `GAN/` module imports both — install them separately when working on the GAN code.

There is no test suite, linter config, or build step. Each script is run directly. Always run from the repository root, because `GAN/train.py`, `GAN/utils.py`, and `GAN/models/*.py` use `sys.path.append(os.getcwd())` to find the top-level `Attractors` and `GAN` packages:

```bash
python -m animations.lorenz_3d        # vispy 3D Lorenz visualization
python animations/lorenz_2d.py        # pygame 2D Lorenz visualization
python animations/random_2d.py        # random 2D quadratic-map attractor search
python GAN/train.py                   # runs TimeGAN training (see note below)
```

Note: `GAN/train.py` executes the full data-loading and training pipeline at **module top level** — there is no `if __name__ == "__main__"` guard. Importing it triggers training. Move new entrypoint logic out of module scope if you need to import from it.

## Architecture

The codebase has three independent layers:

### 1. `Attractors/` — ODE-integrated chaotic systems

`BaseChaoticAttractor` (`Attractors/base_chaotic_attractor.py`) wraps `scipy.integrate.solve_ivp`. Subclasses define the dynamics by overriding `next_state(self, t, state, parameters)` to return derivatives, and expose:

- `DEFAULT_*` class constants for system parameters and initial state.
- An `__init__` that falls back to those defaults when args are `None`, then calls `super().__init__(initial_state, parameters_tuple)`.
- `generate_trajectory(t_span, t_steps)` and `generate_perturbed_trajectories(...)` come for free from the base class.

To add a new attractor:
1. Create `Attractors/<name>.py` subclassing `BaseChaoticAttractor` (use `lorenz.py` as the template).
2. Export it from `Attractors/__init__.py`.
3. Register it in the `attractors` dict in `GAN/utils.py::load_chaotic_data` if it should be available to the GAN pipeline.

### 2. `GAN/` — TimeGAN over chaotic trajectories

A TimeGAN implementation in TensorFlow/Keras. Five Keras `Model` subclasses live in `GAN/models/` and are all built the same way via `GAN/utils.py::make_net`, which stacks `n_layers` of GRU **or** LSTM (selectable per-call via `net_type`) and tops them with a sigmoid `Dense` output:

| Model | Input shape | Output | Role |
|---|---|---|---|
| `Embedder` | `(seq_len, 3)` | `hidden_dim` | real space → latent |
| `Recovery` | `(seq_len, hidden_dim)` | `3` | latent → real space |
| `Generator` | `(seq_len, hidden_dim)` noise | `3` (via extra `TimeDistributed(Dense(3))`) | noise → synthetic series |
| `Supervisor` | `(seq_len, hidden_dim)` | `hidden_dim` | enforces temporal dynamics (2 layers, not 3) |
| `Discriminator` | `(seq_len, 3)` | `1` | real vs. fake |

The hardcoded `3` is the number of state variables (x, y, z) — all attractors are 3D. Changing dimensionality requires updating these output shapes in every model file.

`GAN/utils.py::load_chaotic_data` instantiates the requested attractors with their defaults, generates trajectories (optionally perturbed), and returns an `(n_samples, seq_len, 3)` array. Data is then `MinMaxScaler`-normalized to `[-1, 1]` in `train.py` before training.

`train_timegan` in `GAN/train.py` runs five training steps per batch (Embedder+Recovery joint, Supervisor, Discriminator, Generator) but is **not** a full TimeGAN — it lacks the joint supervised/embedded generator loss and the autoencoder pretraining phase. Treat it as a work-in-progress baseline.

### 3. `animations/` — Standalone visualizers

Three independent scripts, each duplicating its own dynamics inline rather than importing from `Attractors/`:

- `lorenz_3d.py` — vispy + `TurntableCamera`, vectorized Euler step across `num_points` particles with deque-based fading trails.
- `lorenz_2d.py` — pygame, 2D projection with z-based brightness/size depth cue.
- `random_2d.py` — searches for chaotic 2D quadratic maps by random coefficient sampling, classifying via a Lyapunov exponent estimate; rejects converging / diverging / non-chaotic series and renders accepted ones with pygame.

If you change Lorenz behavior, note that the dynamics are also duplicated here — they are independent of `Attractors/lorenz.py`.

## Conventions

- Attractor parameters and initial states are stored as `DEFAULT_*` class constants and packed into a `parameters` tuple passed through `solve_ivp` as `args=(self.parameters,)`. Keep that contract intact — `next_state` unpacks the tuple positionally.
- Imports across packages rely on running from the repo root; do not change a script to be runnable from a subdirectory without updating the `sys.path` shim consistently.
