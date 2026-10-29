import numpy as np
import pytest
from planner_fakes import GridFB

from fbplan.option_model import OptionModel, OptionModelConfig

START, FAR_END = (1, 4), (5, 4)  # the two bottom ends of the U maze, 10 steps apart


class ScaledBackward:
    """Wraps an FBModel so that B(w), and therefore z_w = B(w), is c times larger."""

    def __init__(self, fb: GridFB, c: float) -> None:
        self.fb, self.c = fb, c

    def __getattr__(self, name: str):
        return getattr(self.fb, name)

    def backward(self, obs: np.ndarray) -> np.ndarray:
        return self.c * self.fb.backward(obs)


def all_cells(fb: GridFB) -> np.ndarray:
    return np.stack([fb.obs(c) for c in fb.cells])


def table(fb, z_r=None, **cfg):
    om = OptionModel(fb, OptionModelConfig(**cfg))
    obs = all_cells(fb)
    return om.pair_table(obs, om.targets(obs), z_r)


def test_rho_is_gamma_to_the_hitting_time_on_the_exact_fake():
    fb = GridFB()
    np.testing.assert_allclose(table(fb).rho, fb.discount**fb.dist, rtol=1e-6)


def test_rho_and_reward_do_not_depend_on_the_scale_of_b():
    fb = GridFB(goals={1: (3, 1)})
    z_r = fb.infer_task_latent(1)
    base, scaled = table(fb, z_r), table(ScaledBackward(fb, 37.0), z_r)
    np.testing.assert_allclose(scaled.rho, base.rho, rtol=1e-6)
    np.testing.assert_allclose(scaled.reward, base.reward, rtol=1e-6, atol=1e-9)


def test_exact_reward_counts_only_reward_before_the_hit():
    fb = GridFB(goals={1: (3, 1)})
    z_r = fb.infer_task_latent(1)  # reward 3 at the goal cell
    t = table(fb, z_r, edge_reward="exact")
    i, j, g = fb.index[(1, 1)], fb.index[(5, 1)], fb.index[(3, 1)]
    np.testing.assert_allclose(t.reward[i, j], 3.0 * fb.discount**2)  # passes the goal on the way
    np.testing.assert_allclose(t.reward[i, g], 0.0, atol=1e-9)  # the hit itself is not counted


def test_paper_reward_drops_the_term_after_the_hit():
    fb = GridFB(goals={1: (3, 1)})
    z_r = fb.infer_task_latent(1)
    t = table(fb, z_r, edge_reward="paper")
    i, g = fb.index[(1, 1)], fb.index[(3, 1)]
    np.testing.assert_allclose(t.reward[i, g], 3.0 * fb.discount**2 / (1 - fb.discount))


def test_zero_reward_mode_returns_zeros():
    fb = GridFB()
    t = table(fb, fb.infer_task_latent(1), edge_reward="zero")
    assert not t.reward.any()


def test_no_task_latent_gives_no_reward_table():
    assert table(GridFB()).reward is None


def test_direct_value_is_the_task_value_of_the_task_policy():
    fb = GridFB()
    om = OptionModel(fb, OptionModelConfig())
    obs = all_cells(fb)
    expected = 3.0 * fb.discount ** fb.dist[:, fb.index[FAR_END]] / (1 - fb.discount)
    np.testing.assert_allclose(om.direct_value(obs, fb.infer_task_latent(1)), expected)


@pytest.mark.parametrize(
    ("head_agg", "expected"), [("min", 0.9**10), ("mean", (0.9**10 + 0.95) / 2)]
)
def test_head_aggregation_of_a_one_head_wormhole(head_agg, expected):
    fb = GridFB(wormholes={(1, START, FAR_END): 0.95})
    rho = table(fb, head_agg=head_agg).rho
    np.testing.assert_allclose(rho[fb.index[START], fb.index[FAR_END]], expected, rtol=1e-6)


def test_rho_is_clipped_to_the_unit_interval_and_clips_are_counted():
    wormholes = {(h, START, FAR_END): 1.5 for h in (0, 1)}
    wormholes |= {(h, FAR_END, START): -0.2 for h in (0, 1)}
    fb = GridFB(wormholes=wormholes)
    t = table(fb)
    assert t.rho[fb.index[START], fb.index[FAR_END]] == 1.0
    assert t.rho[fb.index[FAR_END], fb.index[START]] == 0.0
    assert (t.n_clipped_low, t.n_clipped_high) == (1, 1)


def test_targets_with_nonpositive_self_measure_are_unreachable():
    fb = GridFB(wormholes={(h, FAR_END, FAR_END): -1.0 for h in (0, 1)})
    rho = table(fb).rho
    assert not rho[:, fb.index[FAR_END]].any()


def test_batch_size_does_not_change_the_tables():
    fb = GridFB(goals={1: (3, 1)})
    z_r = fb.infer_task_latent(1)
    big, small = table(fb, z_r, batch_size=4096), table(fb, z_r, batch_size=3)
    np.testing.assert_allclose(small.rho, big.rho)
    np.testing.assert_allclose(small.reward, big.reward)


def test_unknown_config_values_are_rejected():
    with pytest.raises(ValueError):
        OptionModelConfig(head_agg="max")
