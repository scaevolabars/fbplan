import dataclasses
import math

import pandas as pd
import pytest

from fbplan.contracts import EpisodeRecord
from fbplan.evaluate import Checkpoint, write_csv
from fbplan.stats import (
    CHECKPOINT_DEFAULTS,
    load_results,
    mcnemar_p_value,
    paired_comparison,
    paired_table,
    pool_label,
    row_labels,
    success_ci,
    summary_table,
    unpaired_comparison,
)


def _episodes(successes_by_task: dict[int, list[int]], method: str = "m", **extra) -> pd.DataFrame:
    rows = [
        {"method": method, "env_name": "env", "task_id": task_id, "eval_seed": 0, "episode": i,
         "success": bool(s), **extra}
        for task_id, successes in successes_by_task.items()
        for i, s in enumerate(successes)
    ]
    return pd.DataFrame(rows)


def test_success_ci_is_a_point_when_every_episode_succeeds():
    est = success_ci(_episodes({1: [1] * 20, 2: [1] * 20}))
    assert (est.mean, est.low, est.high, est.n) == (1.0, 1.0, 1.0, 40)


def test_success_ci_width_matches_binomial_standard_error_for_one_task():
    est = success_ci(_episodes({1: [1, 0] * 200}))
    half_width = (est.high - est.low) / 2
    assert est.mean == 0.5
    assert half_width == pytest.approx(1.96 * math.sqrt(0.25 / 400), rel=0.1)


def test_overall_success_weights_tasks_equally():
    assert success_ci(_episodes({1: [1] * 10, 2: [0] * 30})).mean == 0.5


def test_stratified_ci_has_no_width_from_differences_between_tasks():
    est = success_ci(_episodes({1: [1] * 25, 2: [0] * 25}))
    assert (est.low, est.high) == (0.5, 0.5)


def test_cluster_bootstrap_widens_ci_when_clusters_differ():
    df = pd.concat([
        _episodes({1: [1, 1, 1, 0] * 25}, ckpt=0),
        _episodes({1: [1, 0, 0, 0] * 25}, ckpt=1),
        _episodes({1: [1, 1, 0, 0] * 25}, ckpt=2),
    ])
    plain, clustered = success_ci(df), success_ci(df, cluster="ckpt")
    assert clustered.mean == plain.mean
    assert clustered.high - clustered.low > 1.5 * (plain.high - plain.low)


def test_cluster_bootstrap_rejects_missing_task_in_a_cluster():
    df = pd.concat([_episodes({1: [1], 2: [0]}, ckpt=0), _episodes({1: [1]}, ckpt=1)])
    with pytest.raises(ValueError, match="every task"):
        success_ci(df, cluster="ckpt")


def test_mcnemar_p_value_matches_exact_binomial_tail():
    assert mcnemar_p_value(10, 0) == pytest.approx(2 * 0.5**10)
    assert mcnemar_p_value(3, 5) == pytest.approx(2 * sum(math.comb(8, i) for i in range(4)) / 2**8)
    assert mcnemar_p_value(0, 0) == 1.0
    assert mcnemar_p_value(4, 4) == 1.0


def test_paired_comparison_of_identical_methods_has_no_difference():
    a = _episodes({1: [1, 0, 1, 1], 2: [0, 0, 1, 0]}, method="a")
    result = paired_comparison(pd.concat([a, a.assign(method="b")]), "a", "b")
    assert (result.diff, result.low, result.high, result.p_value) == (0.0, 0.0, 0.0, 1.0)
    assert (result.n_pairs, result.only_a, result.only_b) == (8, 0, 0)


def test_paired_comparison_counts_discordant_pairs_by_method():
    a = _episodes({1: [1] * 12 + [0] * 8}, method="a")
    b = _episodes({1: [0] * 10 + [1] * 2 + [0] * 8}, method="b")
    result = paired_comparison(pd.concat([a, b]), "a", "b")
    assert (result.only_a, result.only_b) == (10, 0)
    assert result.diff == pytest.approx(0.5)
    assert result.p_value == pytest.approx(mcnemar_p_value(10, 0))
    assert result.low > 0.0


def test_paired_comparison_rejects_episodes_that_do_not_pair():
    a = _episodes({1: [1, 1]}, method="a")
    b = _episodes({1: [1, 1, 1]}, method="b")
    with pytest.raises(ValueError, match="pair"):
        paired_comparison(pd.concat([a, b]), "a", "b")


def test_summary_table_has_one_row_per_task_and_one_overall_row_per_method():
    df = pd.concat([_episodes({1: [1, 0], 2: [1, 1]}, "a"), _episodes({1: [0, 0], 2: [1, 0]}, "b")])
    table = summary_table(df)
    assert list(table.columns) == ["method", "task", "n", "success", "ci_low", "ci_high"]
    assert table[["method", "task", "n"]].values.tolist() == [
        ["a", "1", 2], ["a", "2", 2], ["a", "all", 4], ["b", "1", 2], ["b", "2", 2], ["b", "all", 4]
    ]
    assert table.set_index(["method", "task"]).loc[("a", "all"), "success"] == 0.75


def _record(task_id: int, episode: int, success: bool) -> EpisodeRecord:
    return EpisodeRecord("m", "h", "env", task_id, 0, episode, success, 10, 0.1, 0.5)


def test_load_results_concatenates_csv_files(tmp_path):
    write_csv([_record(1, 0, True)], tmp_path / "a.csv")
    write_csv([_record(2, 0, False), _record(2, 1, True)], tmp_path / "b.csv")
    df = load_results([tmp_path / "a.csv", tmp_path / "b.csv"])
    assert df["task_id"].tolist() == [1, 2, 2]
    assert df["success"].dtype == bool


def test_load_results_reads_config_hashes_that_look_like_numbers_as_text(tmp_path):
    hashes = ["58e23249", "0123"]
    records = [dataclasses.replace(_record(1, 0, True), config_hash=h) for h in hashes]
    write_csv(records, tmp_path / "hashes.csv")
    assert load_results([tmp_path / "hashes.csv"])["config_hash"].tolist() == hashes


def test_load_results_with_label_root_names_methods_by_csv_path(tmp_path):
    (tmp_path / "planner").mkdir()
    write_csv([_record(1, 0, True)], tmp_path / "baseline.csv")
    write_csv([_record(1, 0, False)], tmp_path / "planner" / "l0.csv")
    write_csv([_record(1, 0, True)], tmp_path / "planner" / "l1.csv")
    paths = [tmp_path / "baseline.csv", *sorted((tmp_path / "planner").glob("*.csv"))]
    assert load_results(paths)["method"].tolist() == ["m", "m", "m"]
    labelled = load_results(paths, label_root=tmp_path)
    assert labelled["method"].tolist() == ["baseline", "planner/l0", "planner/l1"]


def test_pool_label_drops_the_cluster_part_of_a_row_label():
    labels = ["planner/l0", "baseline@1a2b3c4d", "baseline", "baseline[retrained]/seed3",
              "baseline[b-clean]@1a2b3c4d/seed0"]
    assert [pool_label(x) for x in labels] == [
        "planner", "baseline", "baseline", "baseline[retrained]", "baseline[b-clean]@1a2b3c4d"
    ]


def test_load_results_fills_the_provided_checkpoint_for_csvs_without_one(tmp_path):
    old = pd.DataFrame([dataclasses.asdict(_record(1, 0, True))])
    old.to_csv(tmp_path / "old.csv", index=False)
    write_csv([_record(1, 1, True)], tmp_path / "new.csv", Checkpoint("retrained", "", 4))
    df = load_results([tmp_path / "old.csv", tmp_path / "new.csv"])
    assert df[list(CHECKPOINT_DEFAULTS)].values.tolist() == [
        ["provided", "", -1], ["retrained", "", 4]
    ]


def test_row_labels_name_checkpoints_and_several_configs():
    df = pd.DataFrame({
        "method": ["baseline"] * 4 + ["graph_planner"] * 2,
        "config_hash": ["h0"] * 4 + ["p1", "p2"],
        "checkpoint_kind": ["provided", "retrained", "retrained", "b", "provided", "provided"],
        "checkpoint_variant": ["", "", "", "clean", "", ""],
        "checkpoint_seed": [-1, 0, 1, 2, -1, -1],
    })
    assert row_labels(df).tolist() == [
        "baseline", "baseline[retrained]/seed0", "baseline[retrained]/seed1",
        "baseline[b-clean]/seed2", "graph_planner@p1", "graph_planner@p2",
    ]


def test_summary_table_rejects_results_from_two_envs():
    df = pd.concat([_episodes({1: [1]}), _episodes({1: [0]}).assign(env_name="other")])
    with pytest.raises(ValueError, match="one env"):
        summary_table(df)


def test_paired_table_compares_every_other_method_with_the_reference():
    ref = _episodes({1: [0, 0, 1, 1]}, method="ref")
    better = _episodes({1: [1, 1, 1, 1]}, method="better")
    same = _episodes({1: [0, 0, 1, 1]}, method="same")
    table = paired_table(pd.concat([ref, better, same]), reference="ref")
    assert table["method"].tolist() == ["better", "same"]
    assert table["diff"].tolist() == [0.5, 0.0]
    assert table[["only_method", "only_reference"]].values.tolist() == [[2, 0], [0, 0]]
    assert list(table.columns) == [
        "method", "diff", "ci_low", "ci_high", "only_method", "only_reference", "p_value", "n_pairs"
    ]


def test_unpaired_comparison_of_identical_results_is_centred_on_zero():
    a = _episodes({1: [1, 0] * 50, 2: [1, 1, 1, 0] * 25}, method="a")
    est = unpaired_comparison(pd.concat([a, a.assign(method="b")]), "a", "b")
    assert est.mean == 0.0
    assert est.low < 0.0 < est.high
    assert est.n == 400


def test_unpaired_comparison_detects_a_large_difference():
    a = _episodes({1: [1] * 90 + [0] * 10}, method="a")
    b = _episodes({1: [1] * 50 + [0] * 50}, method="b")
    est = unpaired_comparison(pd.concat([a, b]), "a", "b")
    assert est.mean == pytest.approx(0.4)
    assert est.low > 0.25


def test_unpaired_comparison_rejects_different_task_sets():
    df = pd.concat([_episodes({1: [1], 2: [1]}, method="a"), _episodes({1: [1]}, method="b")])
    with pytest.raises(ValueError, match="same tasks"):
        unpaired_comparison(df, "a", "b")
