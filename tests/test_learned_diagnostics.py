"""Tests for fbplan.learned.diagnostics: the value grid and the rank correlations."""

import jax
import numpy as np

from fbplan.learned.diagnostics import per_goal_correlations, rank_correlation, value_grid
from fbplan.learned.value import ValueConfig, ValueState, multi_switch_value


def test_value_grid_puts_state_i_and_goal_j_at_row_i_column_j(tiny_agent):
    agent = tiny_agent.agent
    d = agent.config["latent_dim"]
    value = ValueState.create(jax.random.PRNGKey(0), tiny_agent.obs_dim, d, ValueConfig((16, 16)))
    rng = np.random.default_rng(0)
    states = rng.normal(size=(3, tiny_agent.obs_dim)).astype(np.float32)
    goals = rng.normal(size=(4, d)).astype(np.float32)
    v1, vh = value_grid(agent, value, states, goals, chunk=5)  # chunks split rows unevenly
    assert v1.shape == vh.shape == (3, 4)
    for i, j in [(0, 0), (1, 3), (2, 1)]:
        e1, eh = multi_switch_value(agent, value, states[i : i + 1], goals[j : j + 1])
        np.testing.assert_allclose([v1[i, j], vh[i, j]], [float(e1[0]), float(eh[0])], rtol=1e-5)


def test_rank_correlations_per_goal_skip_unreachable_pairs():
    geodesic = np.array([[1.0, 2.0], [2.0, np.inf], [3.0, 1.0], [4.0, 3.0]])
    values = -geodesic  # a perfect value falls with the distance
    values[1, 1] = 100.0  # the unreachable pair would spoil goal 1 if it counted
    np.testing.assert_allclose(per_goal_correlations(values, geodesic), [-1.0, -1.0])
    np.testing.assert_allclose(rank_correlation(np.arange(5.0), np.arange(5.0)), 1.0)
