# Report over several harness runs (benchmark/runner.py output folders), built only from their raw files.
# It contains:
# - per-cell means, relative errors and skill against the reference (Seasonal Naive m = 96 at the 28-day context,
#   decision D-B6);
# - Diebold–Mariano tests per sensor;
# - failure counts and the share of negative quantiles;
# - a sensitivity table for the visitor counts with every model's quantiles clipped at 0
#   (D-B3: clipping is never part of the main tables).
#
# All runs must come from the same snapshot, period and harness settings.
# A model scored as a point forecaster (run.json model_params.point_only, e.g. Theta, D-B4) gets no probabilistic
# metrics.
# A failed origin is never imputed. Its relative error and DM test are left empty, with a note.
# With common_origins, every table is built only on the (sensor, origin) pairs that all runs evaluated.
# This lets a model run on an origin subset (origin_stride) be compared with the others on exactly its origins.
# failures.csv still counts each run's own origins.
# Covariate runs (Phase 4) are named "<model>+<set>" and cover one sensor group.
# So the skill is also given per group (halls, weather), next to the main 5-sensor skill.
# dm_base.csv tests each covariate run against its own values-only run ("<model>") on the same origins and context
# (and each widened run against its raw run, dm_base).
# Oracle runs (observed future covariates, run.json covariates.oracle) are upper bounds.
# They get their own section in summary.md, apart from every other model.
#
# Phase 6a (decisions D-E4, D-E6, D-E7; frozen protocol docs/plans/2026-10-22-frozen-protocol.md, thesis root):
# - families.csv: the test families declared in advance (F1-F7). Each test is a DM test of one model against another
#   on per-origin MAE at 24 h (QL as a second column), per sensor, with Holm's correction within the family.
#   Next to it: the effect size, the mean MASE difference at 24 h, with a 95 % interval from a bootstrap over whole
#   Berlin days (the 6-hourly origins overlap, so days, not origins, are resampled).
# - COV80_open: for the halls, the coverage of the 80 % interval on open slots only. A slot is open when the hall's
#   opening profile over the 28 days before the origin is >= 0.3 there (owner decision 2026-10-26; the rule the
#   shipped mask uses, benchmark/models/mask_models.py). The same slots count for every model and context.
#   It needs the snapshot (make_report(snapshot_dir=...)); without it the column is empty.
# - Per-type tables (weather, halls, parking) come first in summary.md; the pooled tables follow.

import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from benchmark.metrics import METRICS, relative_errors, relative_skill, window_metrics
from benchmark.snapshot import BUCKETS_PER_DAY, MANIFEST_NAME, git_state, load_snapshot
from benchmark.tasks import HORIZON, QUANTILE_LEVELS, REPORT_STEPS, TASKS, SensorTask, context_window, target_series

logger = logging.getLogger(__name__)

REFERENCE_LABEL = "reference"
PROBABILISTIC = ("QL", "SQL", "COV80")
DM_HORIZON = 4  # 24-h windows every 6 h overlap 4 origins: HAC lags ≥ 3
QUANTILE_COLUMNS = [f"q{level:g}" for level in QUANTILE_LEVELS]
KEYS = ["model", "sensor", "context", "horizon_steps"]
OPEN_THRESHOLD = 0.3     # a hall slot is open when its 28-day context profile is >= this (owner decision 2026-10-26)
OPEN_CONTEXT_DAYS = 28
ALPHA = 0.05             # Holm within each test family
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 0
BERLIN = "Europe/Berlin"


# One test of a family: model at its context against another model at its context.
@dataclass(frozen=True)
class FamilyTest:
    family: str
    model: str
    context: str
    against: str
    against_context: str


# Reads each run's run.json and metrics.csv.
# Refuses runs that differ in snapshot, period or settings, and two runs of the same model.
# Returns (records by model, metrics of all runs with a "model" column).
def load_runs(run_dirs: list[Path]) -> tuple[dict[str, dict], pd.DataFrame]:
    records, frames, first = {}, [], None
    for run_dir in run_dirs:
        record = json.loads((Path(run_dir) / "run.json").read_text())
        same = (record["snapshot"], record["period"], record["settings"])
        if first is None:
            first = same
        elif same != first:
            raise ValueError(f"RUN {record['run_id']} DIFFERS IN SNAPSHOT, PERIOD OR SETTINGS from the first run")
        if record["model"] in records:
            raise ValueError(f"TWO RUNS OF MODEL {record['model']}")
        record["dir"] = str(run_dir)
        records[record["model"]] = record
        metrics = pd.read_csv(Path(run_dir) / "metrics.csv", parse_dates=["origin"])
        frames.append(metrics.assign(model=record["model"]))
    return records, pd.concat(frames, ignore_index=True)


# Keeps only the metric rows of the (sensor, origin) pairs that every model evaluated. Returns them and the pairs.
def common_origins(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs = metrics[["model", "sensor", "origin"]].drop_duplicates()
    shared = pairs.groupby(["sensor", "origin"])["model"].nunique()
    keep = shared[shared == pairs["model"].nunique()].index.to_frame(index=False)
    return metrics.merge(keep, on=["sensor", "origin"]), keep


# True if the run's model is scored as a point forecaster.
def _point_only(record: dict) -> bool:
    return bool(record.get("model_params", {}).get("point_only", False))


# Mean metrics per model, sensor, context and horizon over the origins that didn't fail, plus the origin counts.
# Probabilistic metrics are NaN for point-only models.
def cells(metrics: pd.DataFrame, records: dict[str, dict]) -> pd.DataFrame:
    ok = metrics[metrics["status"] != "failed"]
    out = ok.groupby(KEYS)[list(METRICS)].mean().reindex(metrics.groupby(KEYS).size().index)
    out["n_origins"] = ok.groupby(KEYS).size()
    out["n_failed"] = metrics[metrics["status"] == "failed"].groupby(KEYS).size()
    out = out.fillna({"n_origins": 0, "n_failed": 0}).reset_index()
    point_only = out["model"].map(lambda m: _point_only(records[m]))
    out.loc[point_only, list(PROBABILISTIC)] = np.nan
    out["point_only"] = point_only
    return out


# Per-origin metric rows of one model, context and horizon, as relative_errors() expects them.
def _rows(metrics: pd.DataFrame, model: str, context: str, h: int, metric: str) -> pd.DataFrame:
    rows = metrics[(metrics["model"] == model) & (metrics["context"] == context) & (metrics["horizon_steps"] == h)]
    return rows[["sensor", "origin", metric]].assign(model=model)


# Relative error of every (model, context) against the reference run at its context, per sensor, horizon and metric
# (MASE, SQL). Origins are paired. A model with failed origins on a sensor gets no relative error, and a note.
def relative(metrics: pd.DataFrame, records: dict[str, dict], reference: tuple[str, int]) -> pd.DataFrame:
    ref_model, ref_days = reference
    if ref_model not in records:
        raise ValueError(f"REFERENCE RUN {ref_model} MISSING")
    out = []
    for metric in ("MASE", "SQL"):
        for h in REPORT_STEPS:
            ref = _rows(metrics, ref_model, f"{ref_days}d", h, metric).assign(model=REFERENCE_LABEL)
            for (model, context), _ in metrics.groupby(["model", "context"], sort=False):
                if metric in PROBABILISTIC and _point_only(records[model]):
                    continue
                own = _rows(metrics, model, context, h, metric)
                for sensor in sorted(own["sensor"].unique()):
                    pair = pd.concat([own[own["sensor"] == sensor], ref[ref["sensor"] == sensor]])
                    row = {"metric": metric, "horizon_steps": h, "model": model, "context": context, "sensor": sensor}
                    try:
                        result = relative_errors(pair, metric, REFERENCE_LABEL).set_index("model").loc[model]
                        out.append({**row, "n_origins": int(result["n_origins"]), "error": result["error"],
                                    "reference_error": result["reference_error"],
                                    "relative_error": result["relative_error"], "note": ""})
                    except ValueError as exc:
                        out.append({**row, "n_origins": int(own.loc[own["sensor"] == sensor, metric].notna().sum()),
                                    "error": np.nan, "reference_error": np.nan, "relative_error": np.nan,
                                    "note": str(exc)})
    return pd.DataFrame(out)


# Sensor groups of the skill: all main sensors (parking is secondary), the main halls and weather stations
# separately, and parking on its own (D-E6: per-type tables).
def sensor_groups(tasks: dict[str, SensorTask]) -> dict[str, list[str]]:
    main = {name: task for name, task in tasks.items() if not task.secondary}
    groups = {"main": sorted(main),
              "halls": sorted(name for name, task in main.items() if task.kind == "visitors"),
              "weather": sorted(name for name, task in main.items() if task.kind == "regular"),
              "parking": sorted(name for name, task in tasks.items() if task.kind == "counter")}
    return {group: sensors for group, sensors in groups.items() if sensors}


# Skill per model, context, horizon, metric and sensor group: the clipped geometric mean of the relative errors over
# the group's sensors. Only cells that have a relative error on every sensor of the group get a value.
def skill(relative_table: pd.DataFrame, tasks: dict[str, SensorTask]) -> pd.DataFrame:
    out = []
    for group, sensors in sensor_groups(tasks).items():
        for (metric, h, model, context), cell in relative_table.groupby(
                ["metric", "horizon_steps", "model", "context"], sort=False):
            cell = cell[cell["sensor"].isin(sensors)]
            complete = cell["relative_error"].notna().sum() == len(sensors)
            row = {"metric": metric, "horizon_steps": h, "model": model, "context": context, "group": group,
                   "n_sensors": len(sensors)}
            if complete:
                result = relative_skill(cell[["model", "relative_error"]]).iloc[0]
                out.append({**row, "gmean_relative_error": result["gmean_relative_error"], "skill": result["skill"],
                            "note": ""})
            else:
                out.append({**row, "gmean_relative_error": np.nan, "skill": np.nan,
                            "note": f"not every {group} sensor has a paired relative error"})
    return pd.DataFrame(out)


# One Diebold–Mariano row for a model's per-origin losses `a` against `b` (both indexed by origin), with `row` as its
# labels. The values are empty, with a note, when the origins don't pair, an origin failed, or the losses are identical.
def _dm_row(a: pd.Series, b: pd.Series, row: dict) -> dict:
    from statsmodels.tsa.api import diebold_mariano_test
    if set(a.index) != set(b.index) or a.isna().any() or b.isna().any():
        return {**row, "statistic": np.nan, "pvalue": np.nan, "lags": np.nan, "mean_loss_difference": np.nan,
                "note": "unpaired or failed origins"}
    b = b.loc[a.index]
    if np.allclose(a.to_numpy(), b.to_numpy()):  # e.g. the same point forecast as the reference
        return {**row, "statistic": np.nan, "pvalue": np.nan, "lags": np.nan,
                "mean_loss_difference": float((a - b).mean()), "note": "identical losses"}
    result = diebold_mariano_test(np.zeros(len(a)), a.to_numpy(), b.to_numpy(), criterion=lambda y, f: f,
                                  horizon=DM_HORIZON, harvey_adj=True)
    return {**row, "statistic": float(result.statistic), "pvalue": float(result.pvalue), "lags": int(result.lags),
            "mean_loss_difference": float((a - b).mean()), "note": ""}


# Diebold–Mariano test of a model against the reference on per-origin losses of one sensor
# (MAE, and QL for probabilistic models).
# It uses Newey–West lags from horizon=DM_HORIZON and the Harvey small-sample correction.
# A negative statistic means the model's loss is lower. Needs statsmodels (bench venv).
def dm_tests(metrics: pd.DataFrame, records: dict[str, dict], reference: tuple[str, int]) -> pd.DataFrame:
    ref_model, ref_days = reference
    out = []
    for loss in ("MAE", "QL"):
        for h in REPORT_STEPS:
            ref = _rows(metrics, ref_model, f"{ref_days}d", h, loss).set_index(["sensor", "origin"])[loss]
            for (model, context), _ in metrics.groupby(["model", "context"], sort=False):
                if (model, context) == (ref_model, f"{ref_days}d"):
                    continue
                if loss == "QL" and _point_only(records[model]):
                    continue
                own = _rows(metrics, model, context, h, loss).set_index(["sensor", "origin"])[loss]
                for sensor in sorted(own.index.get_level_values("sensor").unique()):
                    a, b = own.loc[sensor], ref.loc[sensor]
                    out.append(_dm_row(a, b, {"loss": loss, "horizon_steps": h, "model": model, "context": context,
                                              "sensor": sensor, "n_origins": len(a)}))
    return pd.DataFrame(out)


# Diebold–Mariano tests of each covariate run "<model>+<set>" against its base run, when both are in the report.
# The test is per sensor, loss and horizon, at the same context.
# The base run is the values-only run "<model>", or the run named in model_params.base_model
# (a "0 when closed" run against its unmasked base, with or without a covariate set; a widened run "<run>_w<c>"
# against its raw run "<run>"). A widened run is tested on QL only: widening keeps the median, so its MAE equals the
# raw run's.
# A negative statistic means the covariates (or the mask, or the widening) lower the loss.
def dm_base(metrics: pd.DataFrame, records: dict[str, dict]) -> pd.DataFrame:
    out = []
    for model in records:
        base = records[model].get("model_params", {}).get("base_model") or model.split("+")[0]
        if base == model or base not in records:
            continue
        widened = "band_factor" in records[model].get("model_params", {})
        for loss in ("MAE", "QL"):
            if loss == "QL" and (_point_only(records[model]) or _point_only(records[base])):
                continue
            if loss == "MAE" and widened:
                continue
            for h in REPORT_STEPS:
                for context in metrics.loc[metrics["model"] == model, "context"].unique():
                    own = _rows(metrics, model, context, h, loss).set_index(["sensor", "origin"])[loss]
                    ref = _rows(metrics, base, context, h, loss).set_index(["sensor", "origin"])[loss]
                    for sensor in sorted(own.index.get_level_values("sensor").unique()):
                        if sensor not in ref.index.get_level_values("sensor"):
                            continue
                        a, b = own.loc[sensor], ref.loc[sensor]
                        out.append(_dm_row(a, b, {"loss": loss, "horizon_steps": h, "model": model, "base": base,
                                                  "context": context, "sensor": sensor, "n_origins": len(a)}))
    return pd.DataFrame(out)


# The default families of the frozen protocol that need no choice from validation, for the models in the report:
# - F1: each zero-shot TSFM against the reference;
# - F2: chronos2 against each other zero-shot TSFM;
# - F3: chronos2 against the best statistical baseline sf_mstl_arima (the best tuned trained model is added with
#   extra tests once chosen on validation);
# - F4: chronos2 against each LLMTime model at its own context;
# - F6: chronos2+stations against chronos2, at the long context and at 1 day.
# F5, F7 and F3's trained model name models chosen on validation, so they come as extra tests.
# models: the models in the report; zero_shot, llmtime: registry names with their contexts (days).
def default_families(models: set[str], reference: tuple[str, int], zero_shot: dict[str, tuple[int, ...]],
                     llmtime: dict[str, tuple[int, ...]], days: int = 28) -> list[FamilyTest]:
    context = f"{days}d"
    ref_model, ref_days = reference
    tsfms = [m for m in zero_shot if m in models]
    tests = [FamilyTest("F1", m, context, ref_model, f"{ref_days}d") for m in tsfms]
    if "chronos2" in models:
        tests += [FamilyTest("F2", "chronos2", context, m, context) for m in tsfms if m != "chronos2"]
        if "sf_mstl_arima" in models:
            tests.append(FamilyTest("F3", "chronos2", context, "sf_mstl_arima", context))
        tests += [FamilyTest("F4", "chronos2", context, m, f"{max(c)}d") for m, c in llmtime.items() if m in models]
        if "chronos2+stations" in models:
            tests += [FamilyTest("F6", "chronos2+stations", c, "chronos2", c) for c in (context, "1d")]
    return tests


# Origins grouped by their Berlin calendar day (origins are naive UTC).
def _berlin_days(origins: pd.Index) -> np.ndarray:
    return np.asarray(pd.DatetimeIndex(origins).tz_localize("UTC").tz_convert(BERLIN).date)


# Bootstrap over whole Berlin days of a per-origin series: every resample draws as many days as there are, with
# replacement, and keeps all origins of a drawn day. Returns the mean and its 95 % percentile interval.
def day_bootstrap(values: pd.Series, samples: int = BOOTSTRAP_SAMPLES,
                  seed: int = BOOTSTRAP_SEED) -> tuple[float, float, float]:
    by_day = values.groupby(_berlin_days(values.index))
    sums, counts = by_day.sum().to_numpy(), by_day.size().to_numpy()
    pick = np.random.default_rng(seed).integers(0, len(sums), size=(samples, len(sums)))
    means = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


# Holm's step-down correction of the p-values of one family. The family is declared in advance, so a test that
# couldn't be computed (NaN, e.g. failed origins) still counts: as p = 1, so it can't shrink the family for the others.
# Its own adjusted p-value stays NaN. Returns the adjusted p-values and whether each test is significant at ALPHA.
def holm(pvalues: pd.Series) -> tuple[pd.Series, pd.Series]:
    from statsmodels.stats.multitest import multipletests
    adjusted = pd.Series(np.nan, index=pvalues.index)
    significant = pd.Series(False, index=pvalues.index)
    if len(pvalues):
        reject, corrected, _, _ = multipletests(pvalues.fillna(1.0).to_numpy(), alpha=ALPHA, method="holm")
        known = pvalues.notna().to_numpy()
        adjusted[known], significant[known] = corrected[known], reject[known]
    return adjusted, significant


# The family tests at 24 h, per sensor both models ran on:
# - DM on MAE (the family's test) and on QL (a second column; empty for point-only models);
# - the mean MASE difference (model − against) with its day-bootstrap 95 % interval, and the difference relative to
#   the other model's mean MASE;
# - Holm's correction of the MAE p-values within each family.
# A negative difference or statistic means the model's loss is lower.
def family_tests(metrics: pd.DataFrame, records: dict[str, dict], tests: list[FamilyTest]) -> pd.DataFrame:
    h = max(REPORT_STEPS)
    out = []
    for test in tests:
        if test.model not in records or test.against not in records:
            logger.warning(f"FAMILY TEST SKIPPED, model not in the report: {test}")
            continue
        losses = {}
        for loss in ("MAE", "QL", "MASE"):
            losses[loss] = (_rows(metrics, test.model, test.context, h, loss).set_index(["sensor", "origin"])[loss],
                            _rows(metrics, test.against, test.against_context, h, loss).set_index(
                                ["sensor", "origin"])[loss])
        own, other = losses["MAE"]
        sensors = sorted(set(own.index.get_level_values("sensor")) & set(other.index.get_level_values("sensor")))
        for sensor in sensors:
            label = {"family": test.family, "model": test.model, "context": test.context, "against": test.against,
                     "against_context": test.against_context, "sensor": sensor, "horizon_steps": h}
            row = _dm_row(own.loc[sensor], other.loc[sensor], {**label, "n_origins": len(own.loc[sensor])})
            if _point_only(records[test.model]) or _point_only(records[test.against]):
                row["pvalue_QL"], row["statistic_QL"] = np.nan, np.nan
            else:
                ql = _dm_row(losses["QL"][0].loc[sensor], losses["QL"][1].loc[sensor], {})
                row["pvalue_QL"], row["statistic_QL"] = ql["pvalue"], ql["statistic"]
            a, b = losses["MASE"][0].loc[sensor], losses["MASE"][1].loc[sensor]
            if set(a.index) == set(b.index) and not (a.isna().any() or b.isna().any()):
                diff = a - b.loc[a.index]
                mean, low, high = day_bootstrap(diff)
                row.update({"MASE_difference": mean, "MASE_difference_low": low, "MASE_difference_high": high,
                            "MASE_relative_difference": mean / float(b.mean()),
                            "interval_excludes_0": bool(low > 0 or high < 0)})
            out.append(row)
    table = pd.DataFrame(out)
    if len(table):
        table["pvalue_holm"], table["significant_holm"] = np.nan, False
        for _, rows in table.groupby("family"):
            adjusted, significant = holm(rows["pvalue"])
            table.loc[rows.index, "pvalue_holm"] = adjusted
            table.loc[rows.index, "significant_holm"] = significant
    return table


# Which forecast steps of each hall origin are open: the hall's opening profile over the OPEN_CONTEXT_DAYS before
# the origin is >= OPEN_THRESHOLD in that step's Berlin weekday and slot (the rule of the shipped mask).
# Returns rows (sensor, origin, step, open) for the given (sensor, origin) pairs of visitor sensors.
def open_slots(snapshot_dir: Path, tasks: dict[str, SensorTask], pairs: pd.DataFrame) -> pd.DataFrame:
    from benchmark.models.mask_models import closed_steps
    halls = [name for name, task in tasks.items() if task.kind == "visitors"]
    pairs = pairs[pairs["sensor"].isin(halls)].drop_duplicates()
    if not len(pairs):
        return pd.DataFrame(columns=["sensor", "origin", "step", "open"])
    frames = load_snapshot(Path(snapshot_dir))
    out = []
    for sensor, rows in pairs.groupby("sensor"):
        series = target_series(frames[sensor], tasks[sensor])
        for origin in rows["origin"]:
            context = context_window(series, origin, OPEN_CONTEXT_DAYS * BUCKETS_PER_DAY)
            out.append(pd.DataFrame({"sensor": sensor, "origin": origin, "step": np.arange(1, HORIZON + 1),
                                     "open": ~closed_steps(context, HORIZON, OPEN_THRESHOLD)}))
    return pd.concat(out, ignore_index=True)


# Returns two tables:
# - the share of negative 0.1 and 0.5 quantiles per model, sensor, context and horizon;
# - the metrics of the visitor sensors with every quantile clipped at 0 (a sensitivity table only).
# Reads each run's forecasts.csv.gz. keep: only these (sensor, origin) pairs (None = all).
# opened: open_slots() rows; with them, the negative table also gets COV80_open and n_open_steps for the halls
# (empty for point-only models and without opened).
def forecast_tables(records: dict[str, dict], metrics: pd.DataFrame, tasks: dict[str, SensorTask],
                    keep: Optional[pd.DataFrame] = None,
                    opened: Optional[pd.DataFrame] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    visitors = {name for name, task in tasks.items() if task.kind == "visitors"}
    negative_rows, clipped_rows = [], []
    for model, record in records.items():
        forecasts = pd.read_csv(Path(record["dir"]) / "forecasts.csv.gz", parse_dates=["origin"])
        forecasts = forecasts[forecasts["status"] != "failed"]
        if keep is not None:
            forecasts = forecasts.merge(keep, on=["sensor", "origin"])
        if opened is not None and len(opened):
            forecasts = forecasts.merge(opened, on=["sensor", "origin", "step"], how="left")
            forecasts["covered"] = (forecasts["q0.1"] <= forecasts["y"]) & (forecasts["y"] <= forecasts["q0.9"])
        for h in REPORT_STEPS:
            head = forecasts[forecasts["step"] <= h]
            grouped = head.groupby(["sensor", "context"])
            shares = pd.DataFrame({"negative_q0.1": grouped["q0.1"].apply(lambda q: float((q < 0).mean())),
                                   "negative_q0.5": grouped["q0.5"].apply(lambda q: float((q < 0).mean()))})
            if "open" in head:
                is_open = head[head["open"] == True]  # noqa: E712 (NaN for non-hall rows)
                shares["COV80_open"] = is_open.groupby(["sensor", "context"])["covered"].mean()
                shares["n_open_steps"] = is_open.groupby(["sensor", "context"]).size()
                if _point_only(record):
                    shares["COV80_open"] = np.nan
            negative_rows.append(shares.reset_index().assign(model=model, horizon_steps=h))
        scales = metrics[metrics["model"] == model].drop_duplicates(["sensor", "context", "origin"]).set_index(
            ["sensor", "context", "origin"])["scale"]
        for (sensor, context, origin), window in forecasts[forecasts["sensor"].isin(visitors)].groupby(
                ["sensor", "context", "origin"]):
            window = window.sort_values("step")
            quantiles = np.clip(window[QUANTILE_COLUMNS].to_numpy(dtype=float), 0, None)
            for row in window_metrics(window["y"].to_numpy(dtype=float), quantiles, QUANTILE_LEVELS,
                                      float(scales.loc[(sensor, context, origin)]), REPORT_STEPS):
                clipped_rows.append({"model": model, "sensor": sensor, "context": context, "origin": origin, **row})
    extra = [c for c in ("COV80_open", "n_open_steps") if any(c in r for r in negative_rows)]
    negative = pd.concat(negative_rows, ignore_index=True)[KEYS + ["negative_q0.1", "negative_q0.5", *extra]]
    clipped = pd.DataFrame(clipped_rows)
    if len(clipped):
        clipped = clipped.groupby(KEYS)[list(METRICS)].mean().reset_index()
        point_only = clipped["model"].map(lambda m: _point_only(records[m]))
        clipped.loc[point_only, list(PROBABILISTIC)] = np.nan
    return negative, clipped


# Origin counts per model, sensor and context from each run.json:
# valid and evaluated origins, ok, fallback, failed, and crossed quantile steps.
# With an origin stride fewer origins are evaluated. Runs from before the stride option evaluated every valid origin.
def failures(records: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for model, record in records.items():
        for sensor, counts in record["counts"].items():
            for context, c in counts["contexts"].items():
                rows.append({"model": model, "sensor": sensor, "context": context, "valid_origins": counts["valid"],
                             "evaluated_origins": counts.get("evaluated", counts["valid"]),
                             **{k: c.get(k, 0) for k in ("ok", "fallback", "failed", "crossed_steps")}})
    return pd.DataFrame(rows)


# A Markdown table from a DataFrame, without an extra dependency. Floats get 3 decimals, NaN becomes "n/a".
def _markdown(df: pd.DataFrame) -> str:
    def fmt(v) -> str:
        if isinstance(v, float):
            return "n/a" if math.isnan(v) else f"{v:.3f}"
        return str(v)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


# Sort key: context labels by their number of days ("1d" < "7d" < "14d", "all" last). Other columns stay as they are.
def _context_order(column: pd.Series) -> pd.Series:
    if column.name != "context":
        return column
    return column.map(lambda c: float("inf") if c == "all" else int(c.rstrip("d")))


# True if the run used oracle covariates (observed future values).
def _oracle(record: dict) -> bool:
    return bool((record.get("covariates") or {}).get("oracle", False))


# MASE and SQL per sensor and model at 24 h for each context (only cells that ran), under the given heading prefix.
def _metric_tables(cell_table: pd.DataFrame, title: str) -> list[str]:
    h = max(REPORT_STEPS)
    parts = []
    for metric in ("MASE", "SQL"):
        at_h = cell_table[cell_table["horizon_steps"] == h]
        table = at_h.pivot(index=["model", "context"], columns="sensor", values=metric)  # only cells that ran
        table["n_origins_total"] = at_h.groupby(["model", "context"])["n_origins"].sum().astype(int)
        table = table.reset_index()
        table = table.sort_values(["model", "context"], key=_context_order, ignore_index=True)
        parts.append(f"## {title}{metric} at {h} steps (24 h), mean over origins\n\n{_markdown(table)}\n")
    return parts


# Per-type tables (D-E6, primary): for each sensor group (weather, halls, parking), MASE, SQL and COV80 per sensor at
# 24 h for each model and context, plus the group's MASE skill. For the halls, COV80 is the open-slot coverage
# (COV80_open), or n/a without the snapshot.
def _type_tables(cell_table: pd.DataFrame, skill_table: pd.DataFrame, tasks: dict[str, SensorTask]) -> list[str]:
    h = max(REPORT_STEPS)
    at_h = cell_table[cell_table["horizon_steps"] == h]
    parts = ["# Per-type results (primary, D-E6)\n"]
    for group, sensors in sensor_groups(tasks).items():
        if group == "main":
            continue
        rows = at_h[at_h["sensor"].isin(sensors)]
        if not len(rows):
            continue
        coverage = "COV80_open" if group == "halls" else "COV80"
        if coverage not in rows:
            rows = rows.assign(**{coverage: np.nan})
        table = rows.pivot(index=["model", "context"], columns="sensor", values=["MASE", "SQL", coverage])
        table.columns = [f"{metric} {sensor}" for metric, sensor in table.columns]
        group_skill = skill_table[(skill_table["metric"] == "MASE") & (skill_table["horizon_steps"] == h)
                                  & (skill_table["group"] == group)].set_index(["model", "context"])["skill"]
        table["skill MASE"] = group_skill.reindex(table.index)
        table["n_origins"] = rows.groupby(["model", "context"])["n_origins"].sum().astype(int)
        table = table.reset_index().sort_values(["model", "context"], key=_context_order, ignore_index=True)
        note = " COV80_open: open slots only (28-day context profile >= 0.3)." if group == "halls" else ""
        parts.append(f"## {group}: MASE, SQL and {coverage} at {h} steps (24 h).{note}\n\n{_markdown(table)}\n")
    return parts


# The family tests (Holm within each family): DM on MAE at 24 h and the MASE difference with its day-bootstrap
# interval.
def _family_table(family_table: pd.DataFrame) -> str:
    if not len(family_table):
        return "# Test families\n\nNo family test has both of its models in this report.\n"
    columns = ["family", "model", "context", "against", "against_context", "sensor", "n_origins",
               "mean_loss_difference", "pvalue", "pvalue_holm", "significant_holm", "MASE_difference",
               "MASE_difference_low", "MASE_difference_high", "note"]
    table = family_table[[c for c in columns if c in family_table]]
    return ("# Test families (frozen protocol F1-F7; Holm within each family, alpha 0.05)\n\n"
            "DM on per-origin MAE at 24 h; MASE difference = model − against, 95 % interval from a bootstrap over "
            f"whole Berlin days ({BOOTSTRAP_SAMPLES} resamples). Negative = the model is better.\n\n"
            f"{_markdown(table)}\n")


# Headline tables:
# - the per-type tables and the test families (Phase 6a);
# - MASE and SQL per sensor and model at 24 h for each context;
# - the skill per model, context and sensor group (groups where it exists, and always the main group);
# - the oracle runs in their own section.
def summary_markdown(cell_table: pd.DataFrame, skill_table: pd.DataFrame, records: dict[str, dict],
                     reference: tuple[str, int], common: Optional[int] = None,
                     tasks: Optional[dict[str, SensorTask]] = None,
                     family_table: Optional[pd.DataFrame] = None) -> str:
    first = next(iter(records.values()))
    parts = [f"# Baseline report\n\nSnapshot `{first['snapshot']['id']}`, period `{first['period']}`, "
             f"reference `{reference[0]}` at {reference[1]} days. Generated by `benchmark/report.py`; "
             "point-only models show n/a for probabilistic metrics. Parking is secondary (not in the main skill; its own group).\n"]
    strided = sorted(m for m, r in records.items() if r.get("origin_stride", 1) > 1)
    if strided:
        parts.append(f"**Origin stride:** {', '.join(strided)} evaluated every k-th valid origin. The DM tests keep "
                     f"horizon={DM_HORIZON} (HAC lags for 24-h windows that overlap at 6-h origins); for origins 24 h "
                     "or more apart this is conservative, not invalid.\n")
    if common is not None:
        parts.append(f"**Common origins only:** every table uses the {common} (sensor, origin) pairs that all runs "
                     "evaluated; sensors missing from any run are left out, so the skill may be n/a.\n")
    h = max(REPORT_STEPS)
    oracle = sorted(m for m, r in records.items() if _oracle(r))
    if tasks is not None:
        parts += _type_tables(cell_table[~cell_table["model"].isin(oracle)], skill_table, tasks)
    if family_table is not None:
        parts.append(_family_table(family_table))
    parts.append("# Pooled tables (secondary)\n")
    parts += _metric_tables(cell_table[~cell_table["model"].isin(oracle)], "")
    for metric in ("MASE", "SQL"):
        table = skill_table[(skill_table["metric"] == metric) & (skill_table["horizon_steps"] == h)
                            & ~skill_table["model"].isin(oracle)
                            & ((skill_table["group"] == "main") | skill_table["skill"].notna())].sort_values(
            ["model", "context", "group"], key=_context_order)
        parts.append(f"## Skill on {metric} at 24 h (1 − gmean of relative errors over the group's sensors)\n\n"
                     f"{_markdown(table[['model', 'context', 'group', 'gmean_relative_error', 'skill', 'note']])}\n")
    if oracle:
        parts.append(f"# Oracle runs (upper bounds: observed future covariates)\n\n{', '.join(oracle)}. Not "
                     "comparable with the tables above as forecasts; they bound what perfect covariates could give.\n")
        parts += _metric_tables(cell_table[cell_table["model"].isin(oracle)], "Oracle: ")
    return "\n".join(parts)


# Builds every table from the runs and writes them to out_root/<UTC>_report. Refuses if that folder exists.
# tasks: the sensor tasks the runs used (default: all TASKS).
# common: only the (sensor, origin) pairs every run evaluated (see the header).
# families: the family tests (default_families() plus any chosen on validation); none = no families.csv.
# snapshot_dir: the runs' snapshot, for the open-slot coverage of the halls (refused if its id differs).
# Returns the folder.
def make_report(run_dirs: list[Path], out_root: Path, reference: tuple[str, int],
                tasks: Optional[dict[str, SensorTask]] = None, report_id: Optional[str] = None,
                common: bool = False, families: Optional[list[FamilyTest]] = None,
                snapshot_dir: Optional[Path] = None) -> Path:
    tasks = TASKS if tasks is None else tasks
    started = datetime.now(timezone.utc)
    report_dir = Path(out_root) / (report_id or f"{started:%Y-%m-%dT%H%M%SZ}_report")
    if report_dir.exists():
        raise FileExistsError(f"{report_dir} EXISTS; reports are never overwritten")
    records, metrics = load_runs(run_dirs)
    keep = None
    if common:
        metrics, keep = common_origins(metrics)
    cell_table = cells(metrics, records)
    relative_table = relative(metrics, records, reference)
    skill_table = skill(relative_table, tasks)
    dm_table = dm_tests(metrics, records, reference)
    dm_base_table = dm_base(metrics, records)
    family_table = None if families is None else family_tests(metrics, records, families)
    opened = None
    if snapshot_dir is not None:
        snapshot_id = json.loads((Path(snapshot_dir) / MANIFEST_NAME).read_text())["snapshot_id"]
        if snapshot_id != next(iter(records.values()))["snapshot"]["id"]:
            raise ValueError(f"SNAPSHOT {snapshot_id} IS NOT THE RUNS' SNAPSHOT")
        opened = open_slots(snapshot_dir, tasks, metrics[["sensor", "origin"]])
    negative, clipped = forecast_tables(records, metrics, tasks, keep, opened)
    cell_table = cell_table.merge(negative, on=KEYS, how="left")
    report_dir.mkdir(parents=True)
    cell_table.to_csv(report_dir / "cells.csv", index=False)
    relative_table.to_csv(report_dir / "relative.csv", index=False)
    skill_table.to_csv(report_dir / "skill.csv", index=False)
    dm_table.to_csv(report_dir / "dm.csv", index=False)
    dm_base_table.to_csv(report_dir / "dm_base.csv", index=False)
    clipped.to_csv(report_dir / "clipped_visitors.csv", index=False)
    failures(records).to_csv(report_dir / "failures.csv", index=False)
    if family_table is not None:
        family_table.to_csv(report_dir / "families.csv", index=False)
    n_common = None if keep is None else len(keep)
    (report_dir / "summary.md").write_text(summary_markdown(cell_table, skill_table, records, reference, n_common,
                                                            tasks, family_table))
    first = next(iter(records.values()))
    (report_dir / "report.json").write_text(json.dumps({
        "report_id": report_dir.name, "runs": {m: r["run_id"] for m, r in records.items()},
        "snapshot": first["snapshot"], "period": first["period"], "reference": list(reference),
        "dm": {"horizon": DM_HORIZON, "harvey_adj": True}, "common_origins": n_common,
        "families": None if families is None else [vars(test) for test in families],
        "holm_alpha": ALPHA, "bootstrap": {"samples": BOOTSTRAP_SAMPLES, "seed": BOOTSTRAP_SEED, "unit": "Berlin day"},
        "open_slots": None if snapshot_dir is None else {"context_days": OPEN_CONTEXT_DAYS,
                                                         "threshold": OPEN_THRESHOLD},
        "code": git_state(),
        "created_utc": started.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2))
    logger.info(f"REPORT WRITTEN to {report_dir}")
    return report_dir
