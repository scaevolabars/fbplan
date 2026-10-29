# Brief: a-frozen (checkpoints, env, data)

Read `docs/design.md` first. You are on the critical path: c, d and e need real `FBModel` objects.

## Goal

Implement the stubs in `fbplan/frozen.py` so that the other components can load and query
the frozen models without touching upstream code.

## Owns

`fbplan/frozen.py`, `tests/test_frozen.py`, `docs/status/a-frozen.md`.

## Steps

1. Download all 4 checkpoints (`flags.json`, `params.pkl`) to `checkpoint_dir(env)`.
   Drive folder `1dKYhaDJH9lUREo-kUV3AwmTLrxvKO7Ek`. Known file IDs for medium:
   flags `1r0wR4il8LIbrPKRMakEckCR4MOs7romJ`, params `1683zzDk5v4m2Otad6I2E0TsoYBcGypqB` (18 MB).
   Subfolder listing: `https://drive.google.com/embeddedfolderview?id=<folder-id>`.
   Add a small, idempotent download function or script so the run instructions can reproduce it.
2. The coordinator checked the medium pickle (see design.md §2): numpy arrays only, keys
   `modules_{forward_repr, backward_repr, actor, high_actor}`. Check the other 3 the same way
   (`pickletools` or an unpickler that allows only numpy classes) before a full `pickle.load`.
3. Build the upstream `FBpiSwitchAgent` (`third_party/ssm/agents/fbpiswitch.py`) with
   `get_config()` defaults updated by `flags.json['agent']`, then restore with
   `flax.serialization.from_state_dict`. Import upstream modules through one helper that
   adds `third_party/ssm` to `sys.path`; do not copy upstream code.
4. Implement `FBModel` (see `fbplan/contracts.py`) with jitted batched methods.
   Match upstream exactly: normalization of `z`, both `F` heads, `π_h` input normalization,
   `infer_task_latent` as in upstream `main.py` (relabeled validation data, 100k samples).
5. `make_env` and `load_dataset` as upstream `utils/env_utils.make_env_and_datasets`
   (`_add_noise_to_goal = False` on the eval env). Upstream `main.py` sets `MUJOCO_GL=egl`;
   that fails on macOS. Do not import `main.py`.
6. Sanity check: run upstream `utils/evaluation.evaluate` with the restored agent on medium,
   5 tasks × 50 episodes, 1 seed. Expect about 87 % success. Report the number, the wall time,
   and CPU vs GPU in the status file.

## Tests (`tests/test_frozen.py`)

Shapes and dtypes; `ρ(w,w) = 1` by construction; `z` scale invariance of `forward`;
deterministic `low_action` at temperature 0; `infer_task_latent` is stable across calls.
Skip tests with a clear reason when checkpoints are absent.

## Done when

All 4 checkpoints load, tests pass, the medium sanity number is in the status file, and
`main` can merge your branch. Then tell the user, so the coordinator can merge early.

## Stop and ask the user

Before any download larger than 1 GB (OGBench datasets for large/giant), and if the
sanity number is far from 87 %.
