# Tests for benchmark/report.py on tiny harness runs over a synthetic snapshot (no network):
# - the files, and the reference against itself
# - refusal of mixed snapshots
# - point-only models without probabilistic metrics
# - failed origins left unimputed
# - the negative-quantile share, and the clipped visitor table on a hand example
# - the Diebold–Mariano sign
# - the common-origin report for a run on an origin subset
# - for covariate runs (Phase 4): the skill per sensor group, the DM tests against the values-only run, and the
#   oracle section
# - Phase 6a: Holm's correction on the hand example of the examiner review, the day bootstrap's grouping by Berlin
#   day, the family tests end to end, and the halls' open-slot coverage on a hand example
# The report always runs the DM tests, which need statsmodels. So these tests run in the bench venv only.

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")

from benchmark.bands import widened_forecaster  # noqa: E402
from benchmark.models.mask_models import recent_mask_forecaster  # noqa: E402
from benchmark.report import (FamilyTest, _berlin_days, day_bootstrap, dm_tests, holm,  # noqa: E402
                              make_report)
from benchmark.runner import naive_forecaster, run, seasonal_naive_forecaster  # noqa: E402
from benchmark.snapshot import MANIFEST_NAME, write_raw_csv  # noqa: E402
from benchmark.tasks import REPORT_STEPS, SensorTask  # noqa: E402

TIMES = pd.date_range("2026-01-01", "2026-02-10", freq="15min", inclusive="left")  # 40 days of 15-min rows
PATTERN = np.round(10 + 5 * np.sin(np.arange(96) * 2 * np.pi / 96), 3)


# Writes a snapshot with a weather-like sensor "W" (a daily pattern plus a slow trend, so seasonal naive isn't exact)
# and a visitor counter "V" (always 2 by default), and returns its folder and their tasks (4-day test: 13 origins).
def _snapshot(tmp_path: Path, name: str = "snapshot", shift: float = 0.0,
              visitors: Optional[np.ndarray] = None) -> tuple[Path, dict[str, SensorTask]]:
    snapshot_dir = tmp_path / name
    w = np.tile(PATTERN, len(TIMES) // 96) + np.arange(len(TIMES)) * 0.001 + shift
    v = np.full(len(TIMES), 2.0) if visitors is None else visitors
    files = {"W": pd.DataFrame({"created_at": TIMES, "temperature": w}),
             "V": pd.DataFrame({"created_at": TIMES, "visitors_total": v})}
    sensors = {s: {"file": f"raw/{s}.csv.gz", "sha256": write_raw_csv(df, snapshot_dir / "raw" / f"{s}.csv.gz")}
               for s, df in files.items()}
    (snapshot_dir / MANIFEST_NAME).write_text(json.dumps({"snapshot_id": name, "sensors": sensors}))
    tasks = {"W": SensorTask("temperature", "AVG", "regular", "2026-02-01", "2026-02-05"),
             "V": SensorTask("visitors_total", "MAX", "visitors", "2026-02-01", "2026-02-05")}
    return snapshot_dir, tasks


# Every quantile at -1: a forecaster whose whole interval is below 0.
def _negative_forecaster(contexts, horizon, levels, seed):
    return np.full((len(contexts), horizon, len(levels)), -1.0), ["ok"] * len(contexts)


# Naive forecasts with a NaN at the first origin (marked failed by the runner).
def _failing_forecaster(contexts, horizon, levels, seed):
    quantiles, status = naive_forecaster(contexts, horizon, levels, seed)
    quantiles[0, 0, 0] = np.nan
    return quantiles, status


# Runs a forecaster on the given context lengths (days; default 1) into tmp_path/runs.
def _run(tmp_path: Path, forecaster, model: str, snapshot=None, params=None, context_days=(1,)) -> Path:
    snapshot_dir, tasks = snapshot or _snapshot(tmp_path)
    return run(forecaster, model, snapshot_dir, tmp_path / "runs", tasks=tasks, context_days=context_days,
               run_id=model, model_params=params)


def test_report_files_and_the_reference_against_itself(tmp_path):
    snapshot = _snapshot(tmp_path)
    runs = [_run(tmp_path, seasonal_naive_forecaster(96), "snaive", snapshot),
            _run(tmp_path, naive_forecaster, "naive", snapshot, context_days=(1, 2))]  # models differ in contexts
    report = make_report(runs, tmp_path / "reports", ("snaive", 1), tasks=snapshot[1], report_id="r")

    assert "| snaive | 2d |" not in (report / "summary.md").read_text()  # no rows for cells that never ran
    assert "| naive | 2d |" in (report / "summary.md").read_text()
    for name in ("cells.csv", "relative.csv", "skill.csv", "dm.csv", "clipped_visitors.csv", "failures.csv",
                 "summary.md", "report.json"):
        assert (report / name).exists()
    relative = pd.read_csv(report / "relative.csv")
    own = relative[(relative["model"] == "snaive") & (relative["sensor"] == "W")]
    assert np.allclose(own["relative_error"], 1.0)
    with pytest.raises(FileExistsError):
        make_report(runs, tmp_path / "reports", ("snaive", 1), tasks=snapshot[1], report_id="r")


def test_runs_from_different_snapshots_are_refused(tmp_path):
    first = _run(tmp_path, naive_forecaster, "naive", _snapshot(tmp_path))
    other = _run(tmp_path, seasonal_naive_forecaster(96), "snaive", _snapshot(tmp_path, "other", shift=1.0))
    with pytest.raises(ValueError, match="DIFFERS"):
        make_report([first, other], tmp_path / "reports", ("snaive", 1), tasks=_snapshot(tmp_path)[1])


def test_point_only_models_get_no_probabilistic_metrics(tmp_path):
    snapshot = _snapshot(tmp_path)
    runs = [_run(tmp_path, seasonal_naive_forecaster(96), "snaive", snapshot),
            _run(tmp_path, naive_forecaster, "point", snapshot, params={"point_only": True})]
    report = make_report(runs, tmp_path / "reports", ("snaive", 1), tasks=snapshot[1], report_id="r")
    cells = pd.read_csv(report / "cells.csv")
    point = cells[cells["model"] == "point"]
    assert point[["QL", "SQL", "COV80"]].isna().all().all() and point["MAE"].notna().all()
    relative = pd.read_csv(report / "relative.csv")
    assert not ((relative["model"] == "point") & (relative["metric"] == "SQL")).any()


def test_a_failed_origin_is_reported_not_imputed(tmp_path):
    snapshot = _snapshot(tmp_path)
    runs = [_run(tmp_path, seasonal_naive_forecaster(96), "snaive", snapshot),
            _run(tmp_path, _failing_forecaster, "flaky", snapshot)]
    report = make_report(runs, tmp_path / "reports", ("snaive", 1), tasks=snapshot[1], report_id="r")
    relative = pd.read_csv(report / "relative.csv")
    flaky = relative[(relative["model"] == "flaky") & (relative["sensor"] == "W") & (relative["metric"] == "MASE")]
    assert flaky["relative_error"].isna().all() and flaky["note"].str.contains("UNPAIRED").all()
    assert pd.read_csv(report / "failures.csv").query("model == 'flaky' and sensor == 'W'")["failed"].iloc[0] == 1


def test_negative_share_and_clipped_visitor_metrics_by_hand(tmp_path):
    snapshot = _snapshot(tmp_path)
    runs = [_run(tmp_path, seasonal_naive_forecaster(96), "snaive", snapshot),
            _run(tmp_path, _negative_forecaster, "negative", snapshot)]
    report = make_report(runs, tmp_path / "reports", ("snaive", 1), tasks=snapshot[1], report_id="r")

    cells = pd.read_csv(report / "cells.csv").query("model == 'negative' and sensor == 'V'")
    assert (cells["negative_q0.1"] == 1.0).all() and (cells["MAE"] == 3.0).all()  # y = 2, forecast -1
    clipped = pd.read_csv(report / "clipped_visitors.csv").query("model == 'negative'")
    # Clipped to 0 with y = 2: MAE = 2; QL = mean over deciles of 2·τ·2 = 4·mean(τ) = 2; nothing covered.
    assert set(clipped["sensor"]) == {"V"}
    assert np.allclose(clipped["MAE"], 2.0) and np.allclose(clipped["QL"], 2.0) and (clipped["COV80"] == 0).all()


def test_clipping_changes_nothing_for_non_negative_forecasts(tmp_path):
    # Visitors with closed (0) nights and noise, so the scale is finite and differs per origin and seasonal naive isn't
    # exact. Its forecasts are copies of past values (>= 0), so the clipped table must equal the main one.
    noise = np.random.default_rng(0).normal(0, 3, len(TIMES))
    visitors = np.maximum((np.tile(PATTERN, len(TIMES) // 96) - 10) * 20 + noise, 0)
    snapshot = _snapshot(tmp_path, visitors=visitors)
    runs = [_run(tmp_path, seasonal_naive_forecaster(96), "snaive", snapshot)]
    report = make_report(runs, tmp_path / "reports", ("snaive", 1), tasks=snapshot[1], report_id="r")

    cells = pd.read_csv(report / "cells.csv").query("sensor == 'V'").set_index("horizon_steps")
    clipped = pd.read_csv(report / "clipped_visitors.csv").set_index("horizon_steps")
    assert cells["SQL"].notna().all() and (cells["SQL"] > 0).all()
    for metric in ("MAE", "MASE", "QL", "SQL", "COV80"):
        assert np.allclose(clipped[metric], cells[metric]), metric


def test_dm_statistic_is_negative_for_a_model_with_lower_losses():
    rng = np.random.default_rng(0)
    origins = pd.date_range("2026-02-01", periods=60, freq="6h")
    rows = []
    for model, level in (("ref", 1.0), ("good", 0.5)):
        for h in REPORT_STEPS:
            losses = level + rng.normal(0, 0.1, len(origins))
            rows += [{"model": model, "sensor": "S", "context": "1d", "horizon_steps": h, "origin": o,
                      "MAE": v, "QL": v} for o, v in zip(origins, losses)]
    records = {"ref": {"model_params": {}}, "good": {"model_params": {}}}
    table = dm_tests(pd.DataFrame(rows), records, ("ref", 1)).query("model == 'good'")
    assert len(table) == 2 * len(REPORT_STEPS)
    assert (table["statistic"] < 0).all() and (table["pvalue"] < 0.05).all()



def test_common_origins_score_every_model_on_the_subset_runs_origins(tmp_path):
    # Noisy visitors with closed nights (a finite, origin-specific scale, so MASE pairs exist).
    noise = np.random.default_rng(0).normal(0, 3, len(TIMES))
    snapshot_dir, tasks = _snapshot(tmp_path, visitors=np.maximum((np.tile(PATTERN, len(TIMES) // 96) - 10) * 20
                                                                  + noise, 0))
    full = run(seasonal_naive_forecaster(96), "snaive", snapshot_dir, tmp_path / "runs", tasks=tasks,
               context_days=(1,), run_id="snaive")
    subset = run(_negative_forecaster, "subset", snapshot_dir, tmp_path / "runs", tasks={"V": tasks["V"]},
                 context_days=(1,), run_id="subset", origin_stride=4)  # 4 of 13 origins, visitors only
    report = make_report([full, subset], tmp_path / "reports", ("snaive", 1), tasks=tasks, report_id="r",
                         common=True)

    cells = pd.read_csv(report / "cells.csv")
    assert set(cells["sensor"]) == {"V"} and (cells["n_origins"] == 4).all()
    relative = pd.read_csv(report / "relative.csv")
    assert (relative["n_origins"] == 4).all() and relative["relative_error"].notna().all()
    # The clipped table uses only the 4 common origins: clipped at 0, the subset's MAE over 24 h is the mean of y.
    clipped = pd.read_csv(report / "clipped_visitors.csv").query("model == 'subset' and horizon_steps == 96")
    y = pd.read_csv(subset / "forecasts.csv.gz")["y"]
    assert clipped["MAE"].iloc[0] == pytest.approx(y.mean())
    # The reference ran on all 13 origins; its forecasts are copies of past values (>= 0), so its clipped scores must
    # equal its cells, which use only the 4 common origins.
    ref_clipped = pd.read_csv(report / "clipped_visitors.csv").query("model == 'snaive'").set_index("horizon_steps")
    ref_cells = cells.query("model == 'snaive'").set_index("horizon_steps")
    assert np.allclose(ref_clipped["MAE"], ref_cells["MAE"])
    assert json.loads((report / "report.json").read_text())["common_origins"] == 4
    failures = pd.read_csv(report / "failures.csv").query("model == 'subset'")
    assert (failures["valid_origins"] == 13).all() and (failures["evaluated_origins"] == 4).all()
    assert (failures[["ok", "fallback", "failed"]].sum(axis=1) == failures["evaluated_origins"]).all()
    assert "Common origins only" in (report / "summary.md").read_text()
    assert "**Origin stride:** subset evaluated" in (report / "summary.md").read_text()
    full_report = make_report([full, subset], tmp_path / "reports", ("snaive", 1), tasks=tasks, report_id="all")
    assert (pd.read_csv(full_report / "cells.csv").query("model == 'snaive'")["n_origins"] == 13).all()


# Naive forecasts plus `shift`, for a covariate run (the windows are accepted and ignored).
def _covariate_forecaster(shift: float):
    def forecast(contexts, horizon, levels, seed, covariates):
        quantiles, status = naive_forecaster(contexts, horizon, levels, seed)
        return quantiles + shift, status
    return forecast


def test_group_skill_and_dm_against_the_values_only_run(tmp_path):
    noise = np.random.default_rng(0).normal(0, 3, len(TIMES))
    snapshot_dir, tasks = _snapshot(tmp_path, visitors=np.maximum((np.tile(PATTERN, len(TIMES) // 96) - 10) * 20
                                                                  + noise, 0))
    out = tmp_path / "runs"
    runs = [run(seasonal_naive_forecaster(96), "snaive", snapshot_dir, out, tasks=tasks, context_days=(1,),
                run_id="snaive"),
            run(naive_forecaster, "naive", snapshot_dir, out, tasks=tasks, context_days=(1,), run_id="naive"),
            run(_covariate_forecaster(0.5), "naive+cal", snapshot_dir, out, tasks={"V": tasks["V"]},
                context_days=(1,), run_id="naive+cal", covariates="cal")]
    report = make_report(runs, tmp_path / "reports", ("snaive", 1), tasks=tasks, report_id="r")

    skill = pd.read_csv(report / "skill.csv").query("metric == 'MASE' and horizon_steps == 96").set_index(
        ["model", "group"])
    assert skill.loc[("naive", "main"), "n_sensors"] == 2
    assert skill.loc[[("naive", g) for g in ("main", "halls", "weather")], "skill"].notna().all()
    assert np.isnan(skill.loc[("naive+cal", "main"), "skill"])  # covers the halls only
    assert skill.loc[("naive+cal", "halls"), "skill"] == pytest.approx(
        1 - pd.read_csv(report / "relative.csv").query(
            "metric == 'MASE' and horizon_steps == 96 and model == 'naive+cal'")["relative_error"].iloc[0])
    dm = pd.read_csv(report / "dm_base.csv")
    assert set(dm["model"]) == {"naive+cal"} and set(dm["base"]) == {"naive"} and set(dm["sensor"]) == {"V"}
    assert (dm["n_origins"] == 13).all() and dm["statistic"].notna().all()
    summary = (report / "summary.md").read_text()
    assert "| naive+cal | 1d | halls |" in summary and "Oracle runs" not in summary


# A "0 when closed" run names its unmasked base in model_params.base_model; dm_base pairs it with that run.
def test_dm_base_pairs_a_masked_run_with_its_named_base(tmp_path):
    noise = np.random.default_rng(0).normal(0, 3, len(TIMES))
    snapshot_dir, tasks = _snapshot(tmp_path, visitors=np.maximum((np.tile(PATTERN, len(TIMES) // 96) - 10) * 20
                                                                  + noise, 0))
    out, halls = tmp_path / "runs", {"V": tasks["V"]}
    runs = [run(seasonal_naive_forecaster(96), "snaive", snapshot_dir, out, tasks=tasks, context_days=(1,),
                run_id="snaive"),
            run(naive_forecaster, "naive", snapshot_dir, out, tasks=halls, context_days=(1,), run_id="naive"),
            run(_covariate_forecaster(0.5), "mask0_naive_t0.1+cal_open", snapshot_dir, out, tasks=halls,
                context_days=(1,), run_id="mask", covariates="cal_open", model_params={"base_model": "naive"})]
    dm = pd.read_csv(make_report(runs, tmp_path / "reports", ("snaive", 1), tasks=tasks, report_id="r") / "dm_base.csv")
    assert set(dm["model"]) == {"mask0_naive_t0.1+cal_open"} and set(dm["base"]) == {"naive"}


# A masked run without a covariate set (Phase 5: the profile comes from each context) is paired with its base too.
def test_dm_base_pairs_a_masked_run_without_a_covariate_set(tmp_path):
    noise = np.random.default_rng(0).normal(0, 3, len(TIMES))
    snapshot_dir, tasks = _snapshot(tmp_path, visitors=np.maximum((np.tile(PATTERN, len(TIMES) // 96) - 10) * 20
                                                                  + noise, 0))
    out, halls = tmp_path / "runs", {"V": tasks["V"]}
    forecast, params = recent_mask_forecaster(lambda: (naive_forecaster, {"point_only": True}), 0.5, "naive")
    runs = [run(seasonal_naive_forecaster(96), "snaive", snapshot_dir, out, tasks=tasks, context_days=(1,),
                run_id="snaive"),
            run(naive_forecaster, "naive", snapshot_dir, out, tasks=halls, context_days=(1,), run_id="naive"),
            run(forecast, "mask0_naive_recent_t0.5", snapshot_dir, out, tasks=halls, context_days=(1,),
                run_id="mask", model_params=params)]
    dm = pd.read_csv(make_report(runs, tmp_path / "reports", ("snaive", 1), tasks=tasks, report_id="r") / "dm_base.csv")
    assert set(dm["model"]) == {"mask0_naive_recent_t0.5"} and set(dm["base"]) == {"naive"}
    assert set(dm["loss"]) == {"MAE"} and (dm["n_origins"] == 13).all()


# A widened run ("<run>_w<c>", run_harness --band-factor) is paired with its raw run, on QL only: widening keeps the
# median, so the MAE can't differ.
def test_dm_base_pairs_a_widened_run_with_its_raw_run_on_ql_only(tmp_path):
    noise = np.random.default_rng(0).normal(0, 3, len(TIMES))
    snapshot_dir, tasks = _snapshot(tmp_path, visitors=np.maximum((np.tile(PATTERN, len(TIMES) // 96) - 10) * 20
                                                                  + noise, 0))
    out, halls = tmp_path / "runs", {"V": tasks["V"]}

    def spread(contexts, horizon, levels, seed):
        quantiles, status = naive_forecaster(contexts, horizon, levels, seed)
        return quantiles + (np.asarray(levels) - 0.5) * 10, status

    runs = [run(seasonal_naive_forecaster(96), "snaive", snapshot_dir, out, tasks=tasks, context_days=(1,),
                run_id="snaive"),
            run(spread, "spread", snapshot_dir, out, tasks=halls, context_days=(1,), run_id="spread"),
            run(widened_forecaster(spread, 2.0), "spread_w2", snapshot_dir, out, tasks=halls, context_days=(1,),
                run_id="spread_w2", model_params={"band_factor": 2.0, "base_model": "spread"})]
    dm = pd.read_csv(make_report(runs, tmp_path / "reports", ("snaive", 1), tasks=tasks, report_id="r") / "dm_base.csv")
    assert set(dm["model"]) == {"spread_w2"} and set(dm["base"]) == {"spread"}
    assert set(dm["loss"]) == {"QL"} and dm["statistic"].notna().all()


def test_oracle_runs_get_their_own_section(tmp_path):
    snapshot_dir = tmp_path / "snapshot"
    w = np.tile(PATTERN, len(TIMES) // 96) + np.arange(len(TIMES)) * 0.001
    raw = pd.DataFrame({"created_at": TIMES, "temperature": w, "humidity": 80.0, "global_radiation": 100.0,
                        "pressure_rel": 1000.0, "wind_speed_avg": 2.0})
    sha256 = write_raw_csv(raw, snapshot_dir / "raw" / "W.csv.gz")
    (snapshot_dir / MANIFEST_NAME).write_text(json.dumps(
        {"snapshot_id": "s", "sensors": {"W": {"file": "raw/W.csv.gz", "sha256": sha256}}}))
    tasks = {"W": SensorTask("temperature", "AVG", "regular", "2026-02-01", "2026-02-05")}
    out = tmp_path / "runs"
    runs = [run(seasonal_naive_forecaster(96), "snaive", snapshot_dir, out, tasks=tasks, context_days=(1,),
                run_id="snaive"),
            run(_covariate_forecaster(0.0), "naive+met_oracle", snapshot_dir, out, tasks=tasks, context_days=(1,),
                run_id="oracle", covariates="met_oracle")]
    summary = (make_report(runs, tmp_path / "reports", ("snaive", 1), tasks=tasks, report_id="r")
               / "summary.md").read_text()
    main, oracle = summary.split("# Oracle runs")
    assert "naive+met_oracle" not in main
    assert "| naive+met_oracle | 1d |" in oracle


# The examiner review's example (Phase 4 RQ3, p 0.033 / 0.024 / 0.31): after Holm none is significant at 0.05.
# By hand: sorted 0.024 × 3 = 0.072, 0.033 × 2 = 0.066 → raised to 0.072 (step-down keeps the order), 0.31 × 1.
def test_holm_on_the_examiner_review_example():
    adjusted, significant = holm(pd.Series([0.033, 0.024, 0.31]))
    assert adjusted.tolist() == pytest.approx([0.072, 0.072, 0.31]) and not significant.any()


# A declared test that couldn't be computed counts as p = 1: with it, 0.02 is multiplied by 2, not 1.
def test_holm_keeps_an_uncomputable_test_in_the_family():
    adjusted, significant = holm(pd.Series([0.02, np.nan]))
    assert adjusted.iloc[0] == pytest.approx(0.04) and np.isnan(adjusted.iloc[1]) and significant.tolist() == [
        True, False]


# Origins are naive UTC; days are Berlin days: 23:00 UTC on 31 Jan is 00:00 on 1 Feb in Berlin (UTC+1 in winter).
def test_bootstrap_resamples_whole_berlin_days():
    origins = pd.DatetimeIndex(["2026-01-31 18:00", "2026-01-31 23:00", "2026-02-01 06:00"])
    assert [str(d) for d in _berlin_days(origins)] == ["2026-01-31", "2026-02-01", "2026-02-01"]
    # Every day holds the same four values (0, 0, 0, 8), so every day has mean 2: resampling whole days always gives 2
    # (a zero-width interval), while resampling single origins would spread it.
    values = pd.Series([0.0, 0.0, 0.0, 8.0] * 5, index=pd.date_range("2026-02-01", periods=20, freq="6h"))
    assert day_bootstrap(values) == (2.0, 2.0, 2.0)


# A family test end to end: naive against seasonal naive on both sensors, Holm within the family, the bootstrap
# interval around the MASE difference, and families.csv plus its summary section.
def test_family_tests_with_holm_and_bootstrap_interval(tmp_path):
    snapshot_dir, tasks = _snapshot(tmp_path)
    out = tmp_path / "runs"
    runs = [run(seasonal_naive_forecaster(96), "snaive", snapshot_dir, out, tasks=tasks, context_days=(1,),
                run_id="snaive"),
            run(naive_forecaster, "naive", snapshot_dir, out, tasks=tasks, context_days=(1,), run_id="naive")]
    report = make_report(runs, tmp_path / "reports", ("snaive", 1), tasks=tasks, report_id="r",
                         families=[FamilyTest("F1", "naive", "1d", "snaive", "1d"),
                                   FamilyTest("F9", "missing", "1d", "snaive", "1d")])
    families = pd.read_csv(report / "families.csv").set_index("sensor")
    assert set(families["family"]) == {"F1"} and set(families.index) == {"W", "V"}
    assert families.loc["W", "statistic"] > 0  # naive is worse than seasonal naive on the daily pattern
    w = families.loc["W"]
    assert w["pvalue_holm"] >= w["pvalue"] and w["MASE_difference_low"] <= w["MASE_difference"] <= w[
        "MASE_difference_high"]
    cells = pd.read_csv(report / "cells.csv").set_index(["model", "sensor", "horizon_steps"])
    assert w["MASE_difference"] == pytest.approx(cells.loc[("naive", "W", 96), "MASE"]
                                                 - cells.loc[("snaive", "W", 96), "MASE"])
    summary = (report / "summary.md").read_text()
    assert "# Test families" in summary and "# Per-type results" in summary and "## weather:" in summary


# Holm runs within each family: the same test in a second family of its own isn't corrected for the first family.
def test_holm_is_applied_within_each_family(tmp_path):
    snapshot_dir, tasks = _snapshot(tmp_path)
    out, weather = tmp_path / "runs", {"W": tasks["W"]}
    runs = [run(seasonal_naive_forecaster(96), "snaive", snapshot_dir, out, tasks=weather, context_days=(1,),
                run_id="snaive"),
            run(naive_forecaster, "naive", snapshot_dir, out, tasks=weather, context_days=(1,), run_id="naive"),
            run(_negative_forecaster, "negative", snapshot_dir, out, tasks=weather, context_days=(1,),
                run_id="negative")]
    report = make_report(runs, tmp_path / "reports", ("snaive", 1), tasks=weather, report_id="r",
                         families=[FamilyTest("F1", "naive", "1d", "snaive", "1d"),
                                   FamilyTest("F1", "negative", "1d", "snaive", "1d"),
                                   FamilyTest("F2", "naive", "1d", "snaive", "1d")])
    families = pd.read_csv(report / "families.csv")
    f1 = families[families["family"] == "F1"]
    f2 = families[families["family"] == "F2"].iloc[0]
    assert f2["pvalue_holm"] == pytest.approx(f2["pvalue"], rel=1e-9, abs=0)  # alone in its family (p ~ 1e-14)
    assert (f1["pvalue_holm"] >= f1["pvalue"]).all() and f1["pvalue_holm"].max() > f1["pvalue"].min()


# Halls' open-slot coverage by hand: visitors are 5 in UTC slots 0-47 and 0 in slots 48-95 every day, so the
# 28-day profile marks exactly slots 0-47 open. An all-zero forecast covers every closed step (y = 0) and no open one:
# COV80 = 0.5 over 24 h, COV80_open = 0, with 48 open steps per origin. The weather sensor gets no open-slot value.
def test_open_slot_coverage_by_hand(tmp_path):
    visitors = np.tile(np.r_[np.full(48, 5.0), np.zeros(48)], len(TIMES) // 96)
    snapshot_dir, tasks = _snapshot(tmp_path, visitors=visitors)

    def zeros(contexts, horizon, levels, seed):
        return np.zeros((len(contexts), horizon, len(levels))), ["ok"] * len(contexts)

    out = tmp_path / "runs"
    runs = [run(seasonal_naive_forecaster(96), "snaive", snapshot_dir, out, tasks=tasks, context_days=(1,),
                run_id="snaive"),
            run(zeros, "zeros", snapshot_dir, out, tasks=tasks, context_days=(1,), run_id="zeros")]
    report = make_report(runs, tmp_path / "reports", ("snaive", 1), tasks=tasks, report_id="r",
                         snapshot_dir=snapshot_dir)
    cells = pd.read_csv(report / "cells.csv").set_index(["model", "sensor", "horizon_steps"])
    v = cells.loc[("zeros", "V", 96)]
    assert v["COV80"] == pytest.approx(0.5) and v["COV80_open"] == 0.0 and v["n_open_steps"] == 48 * 13
    assert np.isnan(cells.loc[("zeros", "W", 96), "COV80_open"])
    with pytest.raises(ValueError, match="NOT THE RUNS' SNAPSHOT"):
        make_report(runs, tmp_path / "reports", ("snaive", 1), tasks=tasks, report_id="r2",
                    snapshot_dir=_snapshot(tmp_path, name="other")[0])
