# DSRL-π₀ with a Custom, Slow-Updating Reward Model (LIBERO)

This guide describes how to run DSRL-π₀ on LIBERO when the environment's
sparse success reward is replaced with a **learned reward model** trained from
**user-supplied trajectory scores**.

The training loop becomes:

1. Roll out `K` trajectories with the current SAC noise policy.
2. Score every trajectory with the user-supplied `f(traj) -> float`.
3. Train a reward model `r̂_θ(o, a)` so that
   `Σ_t r̂_θ(o_t, a_t) ≈ f(τ)`.
4. Use `r̂_θ` to label per-(query-step) rewards for those `K` trajectories
   and write them into the replay buffer.
5. Run SAC updates as usual.
6. Repeat from step 1 with the updated policy.

The reward model is updated **once per `K` rollouts**; the SAC policy is
updated many times per rollout — that is what "reward updates less often than
the policy" means here.

---

## What you need to provide

### 1. A scoring function `f`

The framework expects a callable

```python
def score(traj: dict) -> float: ...
```

`traj` is the dict returned by `collect_traj` in
`examples/train_utils_sim.py` (keys: `observations`, `actions`, `rewards`,
`is_success`, `episode_return`, `images`, `env_steps`). The returned float is
the **target trajectory return** the reward model regresses to. Higher =
better, scale does not need to be normalized.

A starter implementation is at `examples/reward_fn.py`. It assumes you have a
**reference trajectory** and want the reward to be the **negative pixel
discrepancy** between each rollout and that reference (so "closer to the
reference" → larger score). Replace `_load_reference()` and
`_pixel_discrepancy()` with your own.

To point the trainer at a different module, pass `--reward_fn module:callable`
— anything importable in the active venv works (e.g.
`my_pkg.scorers:vlm_score`).

> **Cost note.** `f` is called `K` times per outer iteration. Keep `K` small
> if `f` is expensive (VLM calls, simulator replay, big network forward).

### 2. The training-loop hyperparameters

| Flag | Meaning | Default |
|---|---|---|
| `--use_reward_model` | Turn on the learned-reward path (1/0). | 0 |
| `--reward_fn` | Dotted path `module:callable` for `f`. | `examples.reward_fn:score` |
| `--traj_batch_size` | `K` rollouts between reward-model updates. | 8 |
| `--reward_grad_steps` | Reward-model gradient steps per K-trajectory batch. | 200 |
| `--reward_lr` | Reward-model Adam LR. | 3e-4 |
| `--reward_relabel_buffer` | Re-label *all* prior transitions in the buffer after each reward-model update (1/0). | 0 |

When `--use_reward_model=0`, the trainer is byte-identical to the upstream
behaviour (sparse `-1/0` reward from `is_success`).

---

## Setup

Same as the main README, but using the `uv` venv at the repo root:

```bash
source .venv/bin/activate
```

If you have not installed yet:

```bash
git submodule update --init --recursive
uv venv --python 3.11.11 .venv
source .venv/bin/activate
export UV_LINK_MODE=copy

uv pip install -e .
uv pip install -r requirements.txt
uv pip install "jax[cuda12]==0.5.0"
uv pip install -e openpi
uv pip install -e openpi/packages/openpi-client
uv pip install -e LIBERO --config-settings editable_mode=compat
uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install mujoco==3.3.1
```

Set the LIBERO env vars (same as `examples/scripts/run_libero.sh`):

```bash
export DISPLAY=:0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
export OPENPI_DATA_HOME=./openpi
export EXP=./logs/DSRL_pi0_Libero_CustomReward
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

If you go with the default `examples.reward_fn:score` (pixel discrepancy
against a reference), point it at your reference frames before launching:

```bash
export DSRL_REFERENCE_TRAJ_PATH=/path/to/reference_frames.npz   # uint8 (T,H,W,C)
```

---

## Run

```bash
bash examples/scripts/run_libero_reward.sh
```

…which is the standard `run_libero.sh` plus:

```
--use_reward_model 1 \
--reward_fn examples.reward_fn:score \
--traj_batch_size 8 \
--reward_grad_steps 200 \
--reward_lr 3e-4 \
--reward_relabel_buffer 0
```

You should see, in the log stream:

```
[reward] buffered 1/8 trajs (waiting for batch).
...
[reward] f-scores: mean=...  std=...  min=...  max=...
online buffer timesteps length: ...
... (SAC updates as usual) ...
```

W&B will additionally log:

- `reward_model/loss`
- `reward_model/pred_return_mean`, `reward_model/target_return_mean`
- `reward_model/pred_return_std`, `reward_model/target_return_std`
- `reward_model/updates` (counter)
- `reward_model/f_score_mean`, `reward_model/f_score_std`

---

## What changed in the codebase

- **NEW** `jaxrl2/agents/reward_model/{__init__.py, reward_learner.py}` —
  Flax reward model + return-regression update.
- **NEW** `examples/reward_fn.py` — starter `score(traj)` showing the
  reference-trajectory pattern.
- **NEW** `examples/scripts/run_libero_reward.sh` — wrapper.
- **EDIT** `examples/launch_train_sim.py` — adds the new CLI flags.
- **EDIT** `examples/train_sim.py` — builds the `RewardLearner` and resolves
  `--reward_fn` via `importlib`.
- **EDIT** `examples/train_utils_sim.py` —
  - `trajwise_alternating_training_loop` now defers buffer insertion until
    `K` trajectories are in hand, then scores → trains reward model →
    relabels rewards → inserts.
  - `collect_traj` gained `synthesize_sparse_reward` (default `True`); when
    a learned reward is in use we set placeholder zeros and the loop
    overwrites them.
  - New `_relabel_buffer` helper used by `--reward_relabel_buffer=1`.

The SAC agent (`PixelSACLearner`) is **untouched**.

---

## Tradeoffs and known limitations

These are the design decisions made during implementation. If something does
not work, this is the list to revisit:

1. **Loss form: MSE return-regression (not Bradley–Terry).**
   - We regress `Σ_t r̂(o_t, a_t)` to `f(τ)` directly.
   - Chosen because `f` returns a continuous scalar (your pixel-discrepancy
     metric). MSE is the natural loss.
   - **Switch to pairwise (Bradley–Terry) if:** your `f` is only meaningful
     in *relative* terms (very noisy magnitudes, only ordinal information,
     or you'd rather supply preferences `i ≻ j` than scalars). Replace the
     `loss_fn` in `jaxrl2/agents/reward_model/reward_learner.py:_update_step`
     with the pairwise log-likelihood and feed pairs of trajectories instead
     of `(traj, score)`. ~30 lines of changes, no API churn elsewhere.

2. **Online-only reward training (no reward replay buffer).**
   - Each reward-model batch update sees only the latest `K` trajectories
     (and `--reward_grad_steps` passes over them).
   - **Pro:** simple; the reward model tracks the current rollout
     distribution.
   - **Con:** vulnerable to catastrophic forgetting — old trajectories that
     the reward model used to score correctly may drift.
   - **Mitigation already wired in:** `--reward_relabel_buffer 1` re-scores
     every transition in the SAC replay buffer with the latest `r̂_θ` after
     each update. More expensive but stabilises learning. Off by default
     because it doubles the per-batch reward-model cost when the buffer
     is large.
   - **Further mitigation if needed:** add a rolling buffer of past
     `(traj, score)` pairs that the reward model also trains against. Easy
     extension at the call site in `train_utils_sim.py`.

3. **Per-step reward attribution.**
   - We predict per-(query-step) reward and let `Σ_t r̂` match `f(τ)` end-to-
     end. The model is free to pick any per-step assignment that sums to
     the right total.
   - **If credit assignment matters more than total return:** change the
     target to a *per-step* signal you compute from the trajectory, or
     constrain the per-step prediction (e.g. `softplus`, `tanh` bounding,
     non-decreasing potential function).

4. **Discount in regression vs. discount in critic.**
   - The SAC critic uses `discount = variant.discount ** query_freq` per
     transition. The reward regression uses **flat sum** (no time discount)
     to match `f(τ)` as a whole-trajectory scalar.
   - **Tradeoff:** the predicted per-step `r̂` is what the critic sees. Its
     bootstrapped return therefore equals
     `Σ_t (variant.discount^query_freq)^t · r̂_t`, not `Σ_t r̂_t`. For long
     trajectories this can attenuate the signal.
   - **If this hurts:** either lower the effective discount (`--discount`
     close to 1.0 — the default 0.999^20 ≈ 0.98 is mild) or change the
     regression target to a discounted sum.

5. **Mask handling (terminals).**
   - Same convention as upstream: `mask=0` only at the success-terminal
     query step (when `is_success=True`); otherwise `mask=1`. The learned
     reward does not change termination behaviour.
   - **If you want learned-reward-driven termination:** thread it through
     here.

6. **Trajectory padding.**
   - Reward-model training pads each trajectory to `max_traj_len ≈
     max_timesteps / query_freq + 1`. Padding rows are zeroed in the sum
     via `mask`, but the model still does forward passes on padding inputs
     (cheap given LIBERO's 20-step query length). If you increase
     `max_timesteps` substantially this becomes wasteful — switch to a
     scan-over-real-length implementation.

7. **Reward model encoder is **not** shared with the SAC critic.**
   - Two separate parameter sets, two separate optimizers.
   - **Pro:** clean to swap, no target-network bookkeeping concerns.
   - **Con:** ~2× the encoder memory footprint and a second pass over
     pixels.
   - **If memory/compute matters:** share the encoder and stop-gradient
     across heads. Non-trivial because the SAC encoder is updated by the
     critic loss; you'd want a third head fed by a frozen-or-EMA encoder.

8. **`f` blocks the rollout loop.**
   - `score(traj)` is called synchronously for each of the `K` trajectories
     before the reward-model update.
   - **If `f` is slow:** parallelise inside your `f` (batch K trajectories
     at once and short-circuit the `for` loop in `train_utils_sim.py`), or
     dispatch to a pool. Hook is at the `targets = np.array(...)` line in
     `trajwise_alternating_training_loop`.

9. **Initial-eval dependency.**
   - The first SAC eval (`i == 0`) currently fires inside the SAC update
     branch, which only runs once `len(buffer) > start_online_updates`.
     With `K=8`, that means you wait for ≥8 rollouts before the first eval.
     This is the same as upstream behaviour; only the cadence changes.

---

## FAQ

**Q: My `f` actually compares two trajectories — `f(τ_a, τ_b)` returns a
discrepancy.**
Wrap it in a closure that captures the reference. The starter
`examples/reward_fn.py` shows the pattern: load a reference at module-import
time, return `-discrepancy(traj, REFERENCE)` from `score(traj)`. Negate so
that "closer to reference" = higher score.

**Q: How big should `K` (`--traj_batch_size`) be?**
Big enough that f-score variance averages out (8–32 is a reasonable starting
range), small enough that the policy still gets fresh data frequently. Costs
scale linearly with `K`: `f` is called `K` times and the reward model trains
on `K` trajectories per outer step.

**Q: How do I check the reward model is actually learning?**
Watch `reward_model/loss` and the `pred_return_mean / target_return_mean` gap
on W&B. If the loss bottoms out early but the policy still doesn't learn,
your `f` may not be a good signal — sanity-check it on hand-picked good and
bad trajectories.

**Q: Can I keep the env's sparse success reward as an input to `f`?**
Yes — `traj["is_success"]` is in the dict your callable receives. Combine it
with whatever else you want.
