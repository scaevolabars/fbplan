# Report: planning over sequences of intentions with frozen FB representations

Status: skeleton. Sections marked *(pending)* wait for final runs. Notation: `docs/design.md` §3.

## 1. Summary *(pending)*

The result first: success on `antmaze-medium-navigate-v0` of each method against the baseline,
with the paired difference and its 95 % CI; then large, giant and teleport.

## 2. Setting

- Frozen: forward `F(s,z)` (2 heads), backward `B(s)`, low-level policy `π_l(a|s,z)` and the
  baseline high-level policy `π_h(z|s,z_r)` of FB π-Switch (`third_party/ssm`, arXiv 2605.13207).
- `π_h` picks one intention `z` at every step. Its training target looks one switch ahead:
  "follow `π_{B(w)}` until `w`, then the task policy".
- Constraints: no generative model of states or trajectories, no new environment interaction
  for training, offline OGBench data only.

## 3. From one switch to a chain of switches

Option model from FB, the chain formula by the strong Markov property, the graph Bellman
equation over options, the γ-contraction for distinct landmarks, and the error bound for
continuous states. Full derivation and the exact tabular checks: `docs/theory.md`.

## 4. Approach A: landmark graph planner

Landmarks, trusted edges, value iteration, waypoint execution, and the guards against wormholes
with the evidence for each (`docs/design.md` §4.A).

## 5. Approach B: learned multi-switch value

`V_H(s,z)` by expectile regression on chained option targets, then AWR for a new `π_h` with the
baseline network and controller. Variants clean, wide and region (`docs/plans/approach-b.md`).

## 6. Evaluation protocol

Paired start states, 5 tasks × 50 episodes × 5 eval seeds, retrained baselines as a training
seed factor, stratified bootstrap CIs, McNemar tests, and the parity check against upstream
`evaluate` (`docs/design.md` §5).

## 7. Results *(pending)*

### 7.1 Medium (required)
### 7.2 Large
### 7.3 Giant and teleport

## 8. Analysis *(pending)*

- Why the provided checkpoints score below the paper numbers, and what we ruled out.
- Headroom on medium: the privileged oracle and the failure classes (flips are a `π_l` failure).
- FB distance calibration and wormholes on medium and large.
- Why approach A executes its routes worse than `π_h`.
- What the learned `V_H` gains over the one-switch value, by distance.

## 9. What worked and what did not *(pending)*

## 10. Limitations and next steps *(pending)*

## 11. Reproduction

Commands for every table and figure: `README.md`, section "Reproduce".
