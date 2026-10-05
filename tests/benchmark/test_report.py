# Tests for benchmark/report.py on tiny harness runs over a synthetic snapshot (no network): the files, the reference
# against itself, refusal of mixed snapshots, point-only models without probabilistic metrics, failed origins left
# unimputed, the negative-quantile share and the clipped visitor table on a hand example, and the Diebold–Mariano
# sign. The report always runs the DM tests, which need statsmodels: these tests run in the bench venv only.

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")

from benchmark.report import dm_tests, make_report  # noqa: E402
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
    # exact; its forecasts are copies of past values (>= 0), so the clipped table must equal the main one.
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
