# Brief: b-theory (proofs and exact tabular testbed)

Read `docs/design.md` first. Then read the paper (arXiv 2605.13207; LaTeX source at
`https://arxiv.org/src/2605.13207`): sections 3–5 and appendix B.1 (proof of Theorem 1).
`third_party/ssm` has the code only.

## Goal

1. A rigorous derivation of the chained-switch formula and the graph Bellman equation
   (design.md §3), with the assumptions stated: exact hitting of `w`, strong Markov property,
   what changes in continuous states ("hitting" a ball) and in stochastic envs (teleport).
2. An exact tabular maze where every quantity is computed in closed form
   (`M^π = (I − γP_π)^{-1}`), so that we can test the theory and the planner without FB errors.
3. A controlled study of what goes wrong with approximate FB: wormholes, the dropped term,
   and when chaining beats a single switch.

## Owns

`fbplan/tabular/`, `docs/theory.md`, `docs/figures/theory/`, `tests/test_tabular*.py`,
`docs/status/b-theory.md`.

## First deliverable (priority: c-planner needs it)

`fbplan/tabular/fake_fb.py`: a class that implements the `FBModel` protocol on a small grid maze.
Use one-hot or (x, y) observations. Define `π_z` as the optimal policy for reward `r_z`
(with `B` from an exact or truncated factorization of `M`). Offer an exact mode and a
low-rank / noisy mode so that c-planner can test both. Merge it early.

## Steps after that

1. Numeric tests: Theorem 1 and the chained formula equal the exact values to 1e-8;
   value iteration on a landmark graph equals the optimal SMDP value over these options.
2. Study with the low-rank / noisy mode: success of single-switch vs. chained planning
   as a function of goal distance and rank / noise; effect of `ρ_min`, ensemble min, hop cost;
   effect of the dropped term when it repeats over hops.
3. Write `docs/theory.md`: short, exact, with the plots. It becomes a report appendix.

## Done when

Tests pass, `fake_fb.py` is merged, `docs/theory.md` has the derivation and the study plots.

## Stop and ask the user

If the derivation shows that a formula in design.md §3 is wrong or needs an extra assumption.
