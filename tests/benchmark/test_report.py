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
# - statistics v2 (protocol P2): the 28-lag DM column against statsmodels, Holm on it within each family and the
#   bandwidth wording; the 7-day circular block bootstrap on a hand example and on weekly-correlated days; F3 tests
#   Chronos-2 against both statistical baselines (P3)
# - runs merged as one model (--as): the same tables as one run on both sensors; overlapping sensors refused
# - seeds of one config merged as one model: per-origin losses equal the hand average over seeds; the per-seed cells;
#   seed merges that don't add up are refused
# - runs merged from a validation picks file (--picks): a pooled run's extra sensors enter no table
# - selection on validation: one pick per sensor and model family (tuned configs and covariate sets share a family),
#   never an oracle run, and only validation runs
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
from benchmark.metrics import METRICS  # noqa: E402
from benchmark.report import (DM_LONG_LAGS, FamilyTest, _berlin_days, bandwidth_wording,  # noqa: E402
                              block_bootstrap, cells, day_bootstrap, default_families, dm_tests, holm, load_runs,
                              make_report, model_family, picked_runs, validation_selection)
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


# The second DM column (protocol P2): the same test with 28 lags (7 days of 6-hourly origins), equal to a direct
# statsmodels call, in dm.csv as in families.csv.
def test_dm_tests_add_a_28_lag_column_equal_to_statsmodels():
    from statsmodels.tsa.api import diebold_mariano_test
    rng = np.random.default_rng(1)
    origins = pd.date_range("2026-02-01", periods=60, freq="6h")
    losses = {"ref": 1.0 + rng.normal(0, 0.2, len(origins)), "good": 0.9 + rng.normal(0, 0.2, len(origins))}
    rows = [{"model": m, "sensor": "S", "context": "1d", "horizon_steps": h, "origin": o, "MAE": v, "QL": v}
            for m, values in losses.items() for h in REPORT_STEPS for o, v in zip(origins, values)]
    records = {"ref": {"model_params": {}}, "good": {"model_params": {}}}
    row = dm_tests(pd.DataFrame(rows), records, ("ref", 1)).query("model == 'good' and loss == 'MAE'").iloc[0]
    direct = diebold_mariano_test(np.zeros(len(origins)), losses["good"], losses["ref"], criterion=lambda y, f: f,
                                  horizon=4, harvey_adj=True, lags=28)
    assert DM_LONG_LAGS == 28 and row["lags_28"] == 28 and row["lags"] == 4  # primary: max(3, ceil(60^(1/3)))
    assert row["pvalue_28"] == pytest.approx(direct.pvalue, rel=1e-12)
    assert row["statistic_28"] == pytest.approx(direct.statistic, rel=1e-12)
    assert row["pvalue_28"] != pytest.approx(row["pvalue"], rel=1e-6)



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


# By hand: three Berlin calendar days, the middle one without origins (kept as an empty day). A 7-day block spans the
# whole 3-day period, so every circular resample draws each day exactly once: the interval is the mean, (1+3+8)/3.
# The single-day bootstrap spreads the same values.
def test_block_bootstrap_draws_each_day_once_when_a_block_spans_the_period():
    values = pd.Series([1.0, 3.0, 8.0], index=pd.DatetimeIndex(["2026-02-01 06:00", "2026-02-01 12:00",
                                                                "2026-02-03 06:00"]))
    assert block_bootstrap(values, block_days=7) == (4.0, 4.0, 4.0)
    assert day_bootstrap(values)[2] > 4.0


# Loss differences that stay correlated within each week (+1 one week, -1 the next, for 8 weeks): resampling single
# days treats them as independent and gives a narrow interval; 7-day blocks keep the weeks and give a wider one.
# Expected width ratio: a block starting s days into a week averages 1 - 2s/7, whose mean square over s = 0..6 is
# 0.347, so the ratio is about sqrt(7 * 0.347) = 1.56; with 1-day blocks it would be about 1.
def test_block_bootstrap_widens_the_interval_for_weekly_correlated_days():
    origins = pd.date_range("2026-01-05", periods=56 * 4, freq="6h")  # Mondays at 00:00 UTC start each week
    week = np.asarray((origins - origins[0]).days // 7)
    noise = np.random.default_rng(0).normal(0, 0.1, len(origins))
    values = pd.Series(np.where(week % 2 == 0, 1.0, -1.0) + noise, index=origins)
    _, day_low, day_high = day_bootstrap(values)
    mean, low, high = block_bootstrap(values)
    assert mean == pytest.approx(values.mean()) and low <= mean <= high
    assert (high - low) > 1.3 * (day_high - day_low)


# The wording rule of protocol P2: robust only when significant after Holm under both lag settings.
def test_bandwidth_wording():
    assert bandwidth_wording(True, True) == "robust"
    assert bandwidth_wording(True, False) == "depends on the bandwidth"
    assert bandwidth_wording(False, True) == "depends on the bandwidth"
    assert bandwidth_wording(False, False) == "not significant"


# F3 tests Chronos-2 against both statistical baselines when they are in the report (P3 adds ag_savg672).
def test_f3_tests_chronos2_against_both_statistical_baselines():
    zero_shot = {"chronos2": (28,)}
    tests = default_families({"chronos2", "sf_mstl_arima", "ag_savg672", "ag_snaive96"}, ("ag_snaive96", 28),
                             zero_shot, {})
    f3 = [(t.model, t.against, t.context, t.against_context) for t in tests if t.family == "F3"]
    assert f3 == [("chronos2", "sf_mstl_arima", "28d", "28d"), ("chronos2", "ag_savg672", "28d", "28d")]
    without = default_families({"chronos2", "sf_mstl_arima", "ag_snaive96"}, ("ag_snaive96", 28), zero_shot, {})
    assert [t.against for t in without if t.family == "F3"] == ["sf_mstl_arima"]


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


# Statistics v2 end to end: the 28-lag p-values get their own Holm correction within each family, the bandwidth
# wording follows both decisions, the 7-day block interval sits next to the single-day one, and summary.md and
# report.json say so.
def test_family_tests_add_the_28_lag_holm_wording_and_block_interval(tmp_path):
    snapshot_dir, tasks = _snapshot(tmp_path)
    out, weather = tmp_path / "runs", {"W": tasks["W"]}
    runs = [run(seasonal_naive_forecaster(96), "snaive", snapshot_dir, out, tasks=weather, context_days=(1,),
                run_id="snaive"),
            run(naive_forecaster, "naive", snapshot_dir, out, tasks=weather, context_days=(1,), run_id="naive"),
            run(_negative_forecaster, "negative", snapshot_dir, out, tasks=weather, context_days=(1,),
                run_id="negative")]
    report = make_report(runs, tmp_path / "reports", ("snaive", 1), tasks=weather, report_id="r",
                         families=[FamilyTest("F1", "naive", "1d", "snaive", "1d"),
                                   FamilyTest("F1", "negative", "1d", "snaive", "1d")])
    families = pd.read_csv(report / "families.csv")
    adjusted, significant = holm(families["pvalue_28"])
    assert families["pvalue_holm_28"].tolist() == pytest.approx(adjusted.tolist(), rel=1e-12, abs=0)  # p ~ 1e-20
    assert families["significant_holm_28"].tolist() == significant.tolist()
    assert (families["lags_28"] == 28).all()
    assert families["bandwidth"].tolist() == [bandwidth_wording(a, b) for a, b in zip(
        families["significant_holm"], families["significant_holm_28"])]
    for _, row in families.iterrows():
        mean, low, high = row[["MASE_difference", "MASE_difference_low_block7", "MASE_difference_high_block7"]]
        assert low <= mean <= high
        assert row["interval_excludes_0_block7"] == bool(low > 0 or high < 0)
    summary = (report / "summary.md").read_text()
    assert "pvalue_holm_28" in summary and "bandwidth" in summary and "MASE_difference_low_block7" in summary
    record = json.loads((report / "report.json").read_text())
    assert record["dm"]["long_lags"] == 28
    assert record["block_bootstrap"] == {"samples": 2000, "seed": 0, "block_days": 7,
                                         "unit": "Berlin calendar day, circular"}


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


def test_model_family_strips_the_set_and_the_tuned_config():
    assert model_family("ag_deepar_tuned_c672-lr0.001+cal") == "ag_deepar"
    assert model_family("chronos2+cal_open_wx") == "chronos2" and model_family("ag_ets") == "ag_ets"


# Two tuned configs (one with a covariate set) form one family per sensor; the lower SQL wins. An oracle run and a run
# on uncommitted code are never picked, even with the lowest SQL. A test-period run and a missing context are refused.
def test_validation_selection_picks_one_candidate_per_family_and_sensor(tmp_path):
    snapshot_dir, tasks = _snapshot(tmp_path)
    out = tmp_path / "runs"

    def offset(c):  # naive forecasts shifted by c
        def forecast(contexts, horizon, levels, seed):
            quantiles, status = naive_forecaster(contexts, horizon, levels, seed)
            return quantiles + c, status
        return forecast

    def candidate(forecaster, model, run_id, oracle=False, dirty=False):
        run_dir = run(forecaster, model, snapshot_dir, out, tasks=tasks, period="validation", context_days=(1,),
                      run_id=run_id)
        record = json.loads((run_dir / "run.json").read_text())
        record["code"]["dirty"] = dirty  # the test suite may itself run on uncommitted code
        if oracle:
            record["covariates"] = {"oracle": True}  # mark it as an oracle run (its forecasts don't matter here)
        (run_dir / "run.json").write_text(json.dumps(record))
        return run_dir

    runs = [candidate(offset(1.0), "m_tuned_a", "a"), candidate(offset(5.0), "m_tuned_b+cal", "b"),
            candidate(seasonal_naive_forecaster(96), "m+met_oracle", "o", oracle=True),
            candidate(offset(0.0), "m_tuned_d", "d", dirty=True)]
    table = validation_selection(runs, context="1d").set_index(["sensor", "model"])
    assert set(table["family"]) == {"m"}
    assert table.loc[("W", "m_tuned_a"), "selected"] and not table.loc[("W", "m_tuned_b+cal"), "selected"]
    for other in ("m+met_oracle", "m_tuned_d"):
        assert not table.loc[("W", other), "selected"]
        assert table.loc[("W", other), "mean_SQL"] < table.loc[("W", "m_tuned_a"), "mean_SQL"]
    assert not table.loc["V", "selected"].any()  # constant visitors: scale 0, SQL undefined, so no pick
    with pytest.raises(ValueError, match="NO RUN HAS ROWS AT CONTEXT 28d"):
        validation_selection(runs, context="28d")
    test_run = run(offset(0.0), "m_tuned_c", snapshot_dir, out, tasks=tasks, context_days=(1,), run_id="c")
    with pytest.raises(ValueError, match="IS NOT A VALIDATION RUN"):
        validation_selection([test_run], context="1d")


# Two runs of one forecaster on disjoint sensors, merged as "m", give exactly the cells and skill of one run on both
# sensors; merging runs that share a sensor is refused.
def test_runs_merged_as_one_model_equal_one_run_on_all_sensors(tmp_path):
    snapshot_dir, tasks = _snapshot(tmp_path)
    out = tmp_path / "runs"
    ref = run(seasonal_naive_forecaster(96), "snaive", snapshot_dir, out, tasks=tasks, context_days=(1,), run_id="ref")
    whole = run(naive_forecaster, "m", snapshot_dir, out, tasks=tasks, context_days=(1,), run_id="whole")
    w = run(naive_forecaster, "m_w", snapshot_dir, out, tasks={"W": tasks["W"]}, context_days=(1,), run_id="w")
    v = run(naive_forecaster, "m_v", snapshot_dir, out, tasks={"V": tasks["V"]}, context_days=(1,), run_id="v")
    one = make_report([ref, whole], tmp_path / "reports", ("snaive", 1), tasks=tasks, report_id="one")
    merged = make_report([ref, w, v], tmp_path / "reports", ("snaive", 1), tasks=tasks, report_id="merged",
                         aliases={str(w): "m", str(v): "m"})
    for name in ("cells.csv", "skill.csv"):
        pd.testing.assert_frame_equal(pd.read_csv(one / name), pd.read_csv(merged / name))
    assert json.loads((merged / "report.json").read_text())["merged"] == {"m": {"w": "m_w", "v": "m_v"}}
    with pytest.raises(ValueError, match="SHARE SENSORS"):
        make_report([ref, w, whole], tmp_path / "reports", ("snaive", 1), tasks=tasks, report_id="bad",
                    aliases={str(w): "m", str(whole): "m"})
    with pytest.raises(ValueError, match="TWO RUNS OF MODEL m"):  # whole isn't aliased: no silent merge
        make_report([ref, whole, v], tmp_path / "reports", ("snaive", 1), tasks=tasks, report_id="bad2",
                    aliases={str(v): "m"})
    strided = run(naive_forecaster, "m_s", snapshot_dir, out, tasks={"V": tasks["V"]}, context_days=(1,),
                  run_id="s", origin_stride=2)
    with pytest.raises(ValueError, match="ORACLE OR origin_stride"):
        make_report([ref, w, strided], tmp_path / "reports", ("snaive", 1), tasks=tasks, report_id="bad3",
                    aliases={str(w): "m", str(strided): "m"})


# Naive forecasts shifted by c plus 0.1 × seed: each seed of one config scores differently.
def _seeded(c: float):
    def forecast(contexts, horizon, levels, seed):
        quantiles, status = naive_forecaster(contexts, horizon, levels, seed)
        return quantiles + c + 0.1 * seed, status
    return forecast


# Three seeds of one config merged as "m": every per-origin metric is the mean over the seeds' metrics.csv rows, the
# cells and family test are built on those means, seeds.csv holds each seed's own cell means, and failures.csv lists
# every seed's run.
def test_seeds_of_one_config_are_averaged_per_origin(tmp_path):
    snapshot_dir, tasks = _snapshot(tmp_path)
    out = tmp_path / "runs"
    ref = run(seasonal_naive_forecaster(96), "snaive", snapshot_dir, out, tasks=tasks, context_days=(1,), run_id="ref")
    seeds = [run(_seeded(0.5), "m_tuned_a", snapshot_dir, out, tasks=tasks, context_days=(1,), seed=s, run_id=f"a{s}")
             for s in (0, 1, 2)]
    aliases = {str(d): "m" for d in seeds}
    report = make_report([ref, *seeds], tmp_path / "reports", ("snaive", 1), tasks=tasks, report_id="r",
                         aliases=aliases, families=[FamilyTest("F7", "m", "1d", "snaive", "1d")])

    keys = ["sensor", "context", "origin", "horizon_steps"]
    per_seed = pd.concat([pd.read_csv(d / "metrics.csv", parse_dates=["origin"]).assign(seed=s)
                          for s, d in enumerate(seeds)])
    hand = per_seed.groupby(keys)[list(METRICS)].mean()
    _, metrics = load_runs([ref, *seeds], aliases)
    own = metrics[metrics["model"] == "m"].set_index(keys)[list(METRICS)]
    assert len(own) == len(pd.read_csv(seeds[0] / "metrics.csv"))  # one row per origin, not per seed
    pd.testing.assert_frame_equal(own.sort_index(), hand.sort_index())
    assert (own["MAE"] != per_seed[per_seed["seed"] == 0].set_index(keys)["MAE"]).all()  # the seeds do differ

    cells = pd.read_csv(report / "cells.csv").set_index(["model", "sensor", "context", "horizon_steps"])
    hand_cells = hand.groupby(["sensor", "context", "horizon_steps"]).mean()
    for sensor in ("W", "V"):
        for h in REPORT_STEPS:
            cell = cells.loc[("m", sensor, "1d", h)]
            assert cell["MAE"] == pytest.approx(hand_cells.loc[(sensor, "1d", h), "MAE"])
            assert cell["n_origins"] == 13 and cell["n_seeds"] == 3
    assert (cells.loc["snaive", "n_seeds"] == 1).all()

    family = pd.read_csv(report / "families.csv").set_index("sensor").loc["W"]
    ref_mae = pd.read_csv(ref / "metrics.csv").query("sensor == 'W' and horizon_steps == 96")["MAE"].mean()
    assert family["n_origins"] == 13
    assert family["mean_loss_difference"] == pytest.approx(hand.loc["W"].xs(96, level="horizon_steps")["MAE"].mean()
                                                           - ref_mae)

    table = pd.read_csv(report / "seeds.csv")
    assert set(table["model"]) == {"m"} and sorted(table["seed"].unique()) == [0, 1, 2]
    one = table.set_index(["seed", "sensor", "context", "horizon_steps"]).loc[(1, "W", "1d", 96)]
    assert one["MASE"] == pytest.approx(per_seed.query("seed == 1 and sensor == 'W' and horizon_steps == 96")[
        "MASE"].mean())
    assert one["n_origins"] == 13
    seed_mean = table[(table["sensor"] == "W") & (table["horizon_steps"] == 96)]["MASE"].mean()
    assert seed_mean == pytest.approx(cells.loc[("m", "W", "1d", 96), "MASE"])

    failures = pd.read_csv(report / "failures.csv")
    assert sorted(failures.loc[(failures["model"] == "m") & (failures["sensor"] == "W"), "seed"]) == [0, 1, 2]
    assert set(failures.loc[failures["model"] == "m", "run_id"]) == {"a0", "a1", "a2"}
    assert json.loads((report / "report.json").read_text())["seeds"] == {"m": [0, 1, 2]}


# Seed merges that don't add up are refused: one seed twice, seeds keeping different sensors or origins, configs with
# different seed sets, and a kept sensor the run doesn't have.
def test_seed_merges_that_dont_add_up_are_refused(tmp_path):
    snapshot_dir, tasks = _snapshot(tmp_path)
    out = tmp_path / "runs"

    def seed_run(model, seed, run_id, sensors=("W", "V"), context_days=(1,)):
        return run(_seeded(0.5), model, snapshot_dir, out, tasks={s: tasks[s] for s in sensors},
                   context_days=context_days, seed=seed, run_id=run_id)

    a0, a1 = seed_run("m_tuned_a", 0, "a0"), seed_run("m_tuned_a", 1, "a1")
    as_m = lambda *dirs: {str(d): "m" for d in dirs}  # noqa: E731
    with pytest.raises(ValueError, match="SEED 0 OF m_tuned_a TWICE"):
        load_runs([a0, seed_run("m_tuned_a", 0, "a0_again")], as_m(a0, out / "a0_again"))
    with pytest.raises(ValueError, match="KEEP DIFFERENT SENSORS"):
        load_runs([a0, seed_run("m_tuned_a", 2, "a2_w", sensors=("W",))], as_m(a0, out / "a2_w"))
    with pytest.raises(ValueError, match="DIFFER IN ORIGINS"):
        load_runs([a0, seed_run("m_tuned_a", 3, "a3_long", context_days=(1, 2))], as_m(a0, out / "a3_long"))
    b0 = seed_run("m_tuned_b", 0, "b0", sensors=("V",))
    w0, w1 = seed_run("m_tuned_w", 0, "w0", sensors=("W",)), seed_run("m_tuned_w", 1, "w1", sensors=("W",))
    with pytest.raises(ValueError, match="DIFFERENT SEEDS"):
        load_runs([w0, w1, b0], as_m(w0, w1, b0))
    with pytest.raises(ValueError, match="HAS NO SENSORS \\['X'\\]"):
        load_runs([a0, a1], as_m(a0, a1), sensors={str(a0): ["W", "X"], str(a1): ["W"]})


# Writes a picks file like scripts/select_on_validation.py: (sensor, model, family, selected) rows.
def _picks(path: Path, rows: list[tuple[str, str, bool]]) -> Path:
    pd.DataFrame([{"sensor": s, "model": m, "family": model_family(m), "selected": sel} for s, m, sel in rows]).to_csv(
        path, index=False)
    return path


# A pooled config p ran on W and V but was picked only on W; config q was picked on V. Merged from the picks file,
# every table equals the one built from runs of p that never saw V: p's V rows (forecasts below 0, so the negative
# shares and clipped scores would show them) enter no table. Two seeds each.
def test_a_pooled_runs_extra_sensors_enter_no_table(tmp_path):
    snapshot_dir, tasks = _snapshot(tmp_path)
    out = tmp_path / "runs"
    ref = run(seasonal_naive_forecaster(96), "snaive", snapshot_dir, out, tasks=tasks, context_days=(1,), run_id="ref")

    def seed_runs(model, c, sensors, tag):
        return [run(_seeded(c), model, snapshot_dir, out, tasks={s: tasks[s] for s in sensors}, context_days=(1,),
                    seed=s, run_id=f"{tag}{s}") for s in (0, 1)]

    pooled = seed_runs("m_tuned_p_pooled", -5.0, ("W", "V"), "p")
    q = seed_runs("m_tuned_q", 1.0, ("V",), "q")
    w_only = seed_runs("m_tuned_p_pooled", -5.0, ("W",), "pw")
    picks = _picks(tmp_path / "picks.csv", [("W", "m_tuned_p_pooled", True), ("V", "m_tuned_q", True),
                                            ("V", "m_tuned_p_pooled", False), ("W", "other_tuned_x", True)])
    aliases, sensors = picked_runs(picks, "m", [*pooled, *q])
    assert sensors == {str(pooled[0]): ["W"], str(pooled[1]): ["W"], str(q[0]): ["V"], str(q[1]): ["V"]}
    families = [FamilyTest("F7", "m", "1d", "snaive", "1d")]
    picked = make_report([ref, *pooled, *q], tmp_path / "reports", ("snaive", 1), tasks=tasks, report_id="picked",
                         aliases=aliases, sensors=sensors, families=families, snapshot_dir=snapshot_dir,
                         picks={"m": picks})
    clean = make_report([ref, *w_only, *q], tmp_path / "reports", ("snaive", 1), tasks=tasks, report_id="clean",
                        aliases={str(d): "m" for d in [*w_only, *q]}, families=families, snapshot_dir=snapshot_dir)

    for name in ("cells.csv", "relative.csv", "skill.csv", "dm.csv", "families.csv", "clipped_visitors.csv",
                 "seeds.csv"):
        pd.testing.assert_frame_equal(pd.read_csv(picked / name), pd.read_csv(clean / name), obj=name)
    pd.testing.assert_frame_equal(pd.read_csv(picked / "failures.csv").drop(columns="run_id"),
                                  pd.read_csv(clean / "failures.csv").drop(columns="run_id"))
    cells = pd.read_csv(picked / "cells.csv").set_index(["model", "sensor", "context", "horizon_steps"])
    assert cells.loc[("m", "V", "1d", 24), "negative_q0.1"] == 0.0  # p's V forecasts are all below 0
    record = json.loads((picked / "report.json").read_text())
    assert record["picks"]["m"]["file"] == str(picks) and len(record["picks"]["m"]["sha256"]) == 64
    assert record["kept_sensors"] == {"p0": ["W"], "p1": ["W"], "q0": ["V"], "q1": ["V"]}


# A run that isn't a pick, and a pick of the runs' families without a run, are refused.
def test_picked_runs_refuses_unpicked_runs_and_picks_without_runs(tmp_path):
    snapshot_dir, tasks = _snapshot(tmp_path)
    out = tmp_path / "runs"
    a = run(naive_forecaster, "m_tuned_a", snapshot_dir, out, tasks=tasks, context_days=(1,), run_id="a")
    b = run(naive_forecaster, "m_tuned_b", snapshot_dir, out, tasks=tasks, context_days=(1,), run_id="b")
    picks = _picks(tmp_path / "picks.csv", [("W", "m_tuned_a", True), ("V", "m_tuned_b", False)])
    with pytest.raises(ValueError, match="IS NOT A PICK"):
        picked_runs(picks, "m", [a, b])
    picks = _picks(tmp_path / "picks2.csv", [("W", "m_tuned_a", True), ("V", "m_tuned_b", True)])
    with pytest.raises(ValueError, match="PICKS WITHOUT A RUN: \\['m_tuned_b'\\]"):
        picked_runs(picks, "m", [a])


# An origin that failed in one seed is failed in the seed average: empty metrics, counted in n_failed, never imputed
# from the other seeds.
def test_an_origin_failed_in_one_seed_is_failed_in_the_average(tmp_path):
    snapshot_dir, tasks = _snapshot(tmp_path)
    out = tmp_path / "runs"

    def fails_in_seed_1(contexts, horizon, levels, seed):
        return (_failing_forecaster if seed == 1 else naive_forecaster)(contexts, horizon, levels, seed)

    seeds = [run(fails_in_seed_1, "m_tuned_a", snapshot_dir, out, tasks=tasks, context_days=(1,), seed=s,
                 run_id=f"a{s}") for s in (0, 1)]
    records, metrics = load_runs(seeds, {str(d): "m" for d in seeds})
    first = metrics[(metrics["sensor"] == "W") & (metrics["origin"] == metrics["origin"].min())]
    assert (first["status"] == "failed").all() and first[list(METRICS)].isna().all().all()
    cell = cells(metrics, records).set_index(["sensor", "horizon_steps"]).loc[("W", 96)]
    assert cell["n_failed"] == 1 and cell["n_origins"] == 12
