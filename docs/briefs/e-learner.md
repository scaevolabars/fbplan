# Brief: e-learner (GPU: baseline seeds, then approach B)

Read `docs/design.md` first, especially §4.B and §5.

## Goal

Phase 1: a fair baseline with training variance. Phase 2: approach B.

## Owns

`fbplan/train_high.py`, `fbplan/learned/`, `scripts/gpu/`, `tests/test_learned*.py`,
`docs/plans/approach-b.md`, `docs/status/e-learner.md`.

## Phase 1: GPU setup and baseline seeds

1. Ask the user how to reach the GPU machine (SSH host, cloud provider). Write the setup as a
   script in `scripts/gpu/` (clone with submodules, `uv sync` with CUDA JAX, copy checkpoints).
2. `train_high.py`: retrain `π_h` on the frozen `F`, `B`, `π_l` with the upstream recipe
   (`agents/fbpiswitch.py`, 500k steps, `high_alpha=0.1`, same dataset sampling). Load the frozen
   modules through `fbplan.frozen` (a-frozen), not through upstream `load_agent_from_frozen`: the
   provided `params.pkl` is not named `params_<epoch>.pkl` and also holds `modules_high_actor`.
   No wandb by default.
3. Seeds 0–4 on medium, then large. Save to `~/.cache/fbplan/baselines/<env>/seed<k>/`.
   d-eval evaluates them.

## Phase 2: approach B

Write `docs/plans/approach-b.md` for the user to approve before you implement it. It should
cover: target for `V_H(s,z)` (expectile regression on `R(s,w) + ρ(s,w)·V̄_H(w,z)`), how `w` is
sampled, how `z` is sampled (goals and general rewards), the AWR advantage for the new `π_h`,
how to limit error growth over hops, and how B can distill A's routes.

## Done when

Phase 1: 5 retrained `π_h` per env on medium and large, with training curves in `runs/`.
Phase 2: approved plan, implementation with tests, a trained model per env.

## Stop and ask the user

Before every GPU job (cost), and before you implement approach B.
