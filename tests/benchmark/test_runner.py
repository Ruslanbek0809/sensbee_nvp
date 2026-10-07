# Tests for benchmark/runner.py on a tiny synthetic snapshot written to tmp_path (no network): the files and the
# provenance of a run, the check forecasters, determinism by seed, no overwriting, and that failed origins, fallbacks
# and crossed quantiles are counted instead of dropped or hidden.

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmark.runner import CHECK_FORECASTERS, naive_forecaster, run
from benchmark.snapshot import MANIFEST_NAME, sha256_file, write_raw_csv
from benchmark.tasks import QUANTILE_LEVELS, SensorTask

QUANTILE_COLUMNS = [f"q{level:g}" for level in QUANTILE_LEVELS]


TIMES = pd.date_range("2026-01-01", "2026-02-10", freq="15min", inclusive="left")  # 40 days of 15-min rows


# Writes a snapshot with one weather-like sensor "W" (values at TIMES) and returns its folder and a task with a 4-day
# test period: 13 origins, every 6 h.
def _write_snapshot(tmp_path: Path, values: np.ndarray) -> tuple[Path, dict[str, SensorTask]]:
    snapshot_dir = tmp_path / "snapshot"
    sha256 = write_raw_csv(pd.DataFrame({"created_at": TIMES, "temperature": values}), snapshot_dir / "raw" / "W.csv.gz")
    manifest = {"snapshot_id": "test-snapshot", "sensors": {"W": {"file": "raw/W.csv.gz", "sha256": sha256}}}
    (snapshot_dir / MANIFEST_NAME).write_text(json.dumps(manifest))
    return snapshot_dir, {"W": SensorTask("temperature", "AVG", "regular", "2026-02-01", "2026-02-05")}


# A snapshot whose daily pattern repeats exactly.
def _snapshot(tmp_path: Path) -> tuple[Path, dict[str, SensorTask]]:
    pattern = np.round(10 + 5 * np.sin(np.arange(96) * 2 * np.pi / 96), 3)
    return _write_snapshot(tmp_path, np.tile(pattern, len(TIMES) // 96))


# Reads a run's run.json.
def _record(run_dir: Path) -> dict:
    return json.loads((run_dir / "run.json").read_text())


# Random sorted quantiles drawn from the seed: a stand-in for a stochastic model.
def _random_forecaster(contexts, horizon, levels, seed):
    rng = np.random.default_rng(seed)
    return np.sort(rng.normal(size=(len(contexts), horizon, len(levels))), axis=-1), ["ok"] * len(contexts)


# Naive forecasts that misbehave on the origin `bad` ("nan": one NaN quantile, "raise": an exception) and report a
# fallback on the origin 6 h later.
def _flaky_forecaster(bad: pd.Timestamp, mode: str):
    def forecast(contexts, horizon, levels, seed):
        quantiles, status = naive_forecaster(contexts, horizon, levels, seed)
        for i, context in enumerate(contexts):
            origin = context.index[-1] + pd.Timedelta("15min")
            if origin == bad and mode == "raise":
                raise RuntimeError("model crashed")
            if origin == bad:
                quantiles[i, 3, 2] = np.nan
            if origin == bad + pd.Timedelta("6h"):
                status[i] = "fallback"
        return quantiles, status
    return forecast


# Naive forecasts with the quantiles in descending order (every step crossed).
def _reversed_forecaster(contexts, horizon, levels, seed):
    quantiles, status = naive_forecaster(contexts, horizon, levels, seed)
    return quantiles + np.arange(len(levels))[::-1], status


def test_run_writes_raw_results_and_provenance(tmp_path):
    snapshot_dir, tasks = _snapshot(tmp_path)
    run_dir = run(naive_forecaster, "naive", snapshot_dir, tmp_path / "runs", tasks=tasks, context_days=(1, 7),
                  run_id="r1", model_params={"note": "check"})

    forecasts = pd.read_csv(run_dir / "forecasts.csv.gz")
    assert list(forecasts.columns) == ["sensor", "context", "origin", "step", "timestamp", "status", "y",
                                       *QUANTILE_COLUMNS]
    assert len(forecasts) == 13 * 2 * 96  # origins × contexts × steps
    assert len(pd.read_csv(run_dir / "metrics.csv")) == 13 * 2 * 3  # origins × contexts × horizons
    origins = pd.read_csv(run_dir / "origins.csv")
    assert len(origins) == 13 and origins["valid"].all()

    record = _record(run_dir)
    assert record["run_id"] == "r1" and record["model"] == "naive" and record["seed"] == 0
    assert record["model_params"] == {"note": "check"}
    assert record["snapshot"] == {"id": "test-snapshot",
                                  "manifest_sha256": sha256_file(snapshot_dir / MANIFEST_NAME)}
    assert record["forecasts_sha256"] == sha256_file(run_dir / "forecasts.csv.gz")
    assert record["counts"]["W"]["contexts"]["7d"] == {"ok": 13, "fallback": 0, "failed": 0, "crossed_steps": 0}
    assert record["counts"]["W"]["undefined_scale"] == 13  # an exactly periodic series has a zero seasonal scale
    assert record["settings"]["quantile_levels"] == list(QUANTILE_LEVELS)
    assert {"commit", "dirty"} <= set(record["code"]) and {"numpy", "pandas"} <= set(record["versions"])
    assert record["validity_context_days"] == [1, 7, 14, 28]  # the default rule, not stored in settings
    assert "validity_context_days" not in record["settings"]


# A run with a context longer than CONTEXT_DAYS checks that length too: 33 days need history from 2025-12-31 for the
# first origins, which the series doesn't have, so only the 5 origins from 2026-02-03 00:00 on stay valid.
def test_a_longer_run_context_is_part_of_the_validity_rule(tmp_path):
    snapshot_dir, tasks = _snapshot(tmp_path)
    record = _record(run(naive_forecaster, "naive", snapshot_dir, tmp_path / "runs", tasks=tasks, context_days=(1, 33)))

    assert record["validity_context_days"] == [1, 7, 14, 28, 33]
    assert record["counts"]["W"]["valid"] == 5
    assert record["counts"]["W"]["reasons"] == {"ok": 5, "short_history": 8}


def test_mase_and_sql_use_the_28_days_before_each_origin(tmp_path):
    # A ramp whose slope triples on 2026-01-20, inside every origin's 28-day window: each origin gets its own scale,
    # and a window shifted past the origin, or one shared scale, would give different values.
    values = np.cumsum(np.where(TIMES < pd.Timestamp("2026-01-20"), 0.01, 0.03))
    snapshot_dir, tasks = _write_snapshot(tmp_path, values)
    run_dir = run(naive_forecaster, "naive", snapshot_dir, tmp_path / "runs", tasks=tasks, context_days=(1, None))
    metrics = pd.read_csv(run_dir / "metrics.csv", parse_dates=["origin"])

    series = pd.Series(values, index=TIMES)
    scales = metrics.groupby("origin")["scale"].first()
    for origin, scale in scales.items():
        history = series.loc[origin - pd.Timedelta(days=28): origin - pd.Timedelta("15min")].to_numpy()
        assert scale == pytest.approx(np.mean(np.abs(history[96:] - history[:-96])))
    assert scales.nunique() == 13
    assert np.allclose(metrics["MASE"], metrics["MAE"] / metrics["scale"])
    assert np.allclose(metrics["SQL"], metrics["QL"] / metrics["scale"])

    steps = metrics.drop_duplicates(["context", "origin"]).set_index(["context", "origin"])["context_steps"]
    assert (steps.loc["1d"] == 96).all()
    assert all(n == (origin - TIMES[0]) // pd.Timedelta("15min") for origin, n in steps.loc["all"].items())


def test_check_forecasters_on_an_exactly_periodic_series(tmp_path):
    snapshot_dir, tasks = _snapshot(tmp_path)
    seasonal, _ = CHECK_FORECASTERS["seasonal_naive_96"]
    exact = pd.read_csv(run(seasonal, "seasonal_naive_96", snapshot_dir, tmp_path / "a", tasks=tasks,
                            context_days=(1,)) / "metrics.csv")
    assert (exact["MAE"] == 0).all() and (exact["QL"] == 0).all() and (exact["COV80"] == 1).all()

    naive = pd.read_csv(run(naive_forecaster, "naive", snapshot_dir, tmp_path / "b", tasks=tasks,
                            context_days=(1,)) / "metrics.csv")
    assert (naive["MAE"] > 0).all()
    assert np.allclose(naive["QL"], naive["MAE"])  # a point forecast used as every quantile: QL = MAE


def test_same_seed_gives_identical_bytes_and_another_seed_does_not(tmp_path):
    snapshot_dir, tasks = _snapshot(tmp_path)
    hashes = [_record(run(_random_forecaster, "random", snapshot_dir, tmp_path / folder, tasks=tasks,
                          context_days=(1,), seed=seed))["forecasts_sha256"]
              for folder, seed in (("a", 1), ("b", 1), ("c", 2))]
    assert hashes[0] == hashes[1] != hashes[2]


def test_an_existing_run_folder_is_refused(tmp_path):
    snapshot_dir, tasks = _snapshot(tmp_path)
    (tmp_path / "runs" / "r1").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        run(naive_forecaster, "naive", snapshot_dir, tmp_path / "runs", tasks=tasks, context_days=(1,), run_id="r1")


@pytest.mark.parametrize("mode", ["nan", "raise"])
def test_failed_and_fallback_origins_are_counted_not_dropped(tmp_path, mode):
    snapshot_dir, tasks = _snapshot(tmp_path)
    bad = pd.Timestamp("2026-02-02 06:00")
    run_dir = run(_flaky_forecaster(bad, mode), "flaky", snapshot_dir, tmp_path / "runs", tasks=tasks,
                  context_days=(1,))

    assert _record(run_dir)["counts"]["W"]["contexts"]["1d"] == {"ok": 11, "fallback": 1, "failed": 1,
                                                                 "crossed_steps": 0}
    forecasts = pd.read_csv(run_dir / "forecasts.csv.gz", parse_dates=["origin"])
    failed = forecasts[forecasts["origin"] == bad]
    assert len(failed) == 96 and (failed["status"] == "failed").all() and failed[QUANTILE_COLUMNS].isna().all().all()
    metrics = pd.read_csv(run_dir / "metrics.csv", parse_dates=["origin"])
    assert metrics.loc[metrics["origin"] == bad, "MAE"].isna().all()
    assert metrics.loc[metrics["origin"] != bad, "MAE"].notna().all()
    assert (metrics.loc[metrics["origin"] == bad + pd.Timedelta("6h"), "status"] == "fallback").all()


def test_crossed_quantiles_are_counted_then_sorted(tmp_path):
    snapshot_dir, tasks = _snapshot(tmp_path)
    run_dir = run(_reversed_forecaster, "reversed", snapshot_dir, tmp_path / "runs", tasks=tasks, context_days=(1,))

    assert _record(run_dir)["counts"]["W"]["contexts"]["1d"]["crossed_steps"] == 13 * 96
    quantiles = pd.read_csv(run_dir / "forecasts.csv.gz")[QUANTILE_COLUMNS].to_numpy()
    assert (np.diff(quantiles, axis=1) > 0).all()
