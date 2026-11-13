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
# - Statistics v2 (protocol P2, audit F-03): loss differences stay correlated from day to day, so every DM test in
#   dm.csv and dm_base.csv, and the MAE test in families.csv, has a second column with 28 lags (7 days of origins)
#   next to the primary lag rule. A family test is "robust" only if it is significant after Holm under both
#   ("bandwidth" column). The effect-size interval comes from a circular block bootstrap over 7 consecutive Berlin
#   calendar days ("_block7" columns); the single-day interval stays as the second one.
# - COV80_open: for the halls, the coverage of the 80 % interval on open slots only. A slot is open when the hall's
#   opening profile over the 28 days before the origin is >= 0.3 there (owner decision 2026-10-26; the rule the
#   shipped mask uses, benchmark/models/mask_models.py). The same slots count for every model and context.
#   It needs the snapshot (make_report(snapshot_dir=...)); without it the column is empty.
# - Per-type tables (weather, halls, parking) come first in summary.md; the pooled tables follow.
# - Trained models with several seeds (experiments.md: >= 5 seeds on test): the seeds' per-origin losses are averaged
#   before every table and test; seeds.csv holds each seed's own cells (the spread).
# - A pooled run (fitted on every sensor of its kind) keeps only the sensors that picked it on validation
#   (picked_runs); its other sensors enter no table.

import hashlib
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
DM_LONG_LAGS = 28  # the second DM column: 7 days of 6-hourly origins (protocol P2)
BLOCK_DAYS = 7     # block length of the effect-size bootstrap, in calendar days (protocol P2)
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
# aliases (run folder → model name) reports several runs as one model (Phase 6a). Every merged run must be listed in
# aliases and agree on point_only, the oracle flag and the origin stride. Merged runs are either
# - different configs on disjoint sensors, e.g. a tuned model whose configs were picked per sensor on validation, or
# - seeds of one config (same run model name, distinct seeds) on the same sensors and origins. Their per-origin metrics
#   are averaged over the seeds (average_seeds), so every table sees one loss per origin.
# Every config of a merged model must have the same seeds (a forgotten seed folder is refused).
# sensors (run folder → sensors) keeps only those sensors of a run, e.g. the sensors that picked a pooled run; the
# others are dropped from its metrics, counts and forecasts. A kept sensor the run doesn't have is refused.
# The merged record lists every source run ("sources"), folder ("dirs"), source run with its seed and kept counts
# ("runs") and the model's seeds ("seeds").
# Returns (records by model, metrics with a "model" column, averaged over seeds).
def load_runs(run_dirs: list[Path], aliases: Optional[dict[str, str]] = None,
              sensors: Optional[dict[str, list[str]]] = None) -> tuple[dict[str, dict], pd.DataFrame]:
    aliases = {str(Path(k)): v for k, v in (aliases or {}).items()}
    sensors = {str(Path(k)): v for k, v in (sensors or {}).items()}
    records, frames, first = {}, [], None
    for run_dir in run_dirs:
        record = json.loads((Path(run_dir) / "run.json").read_text())
        same = (record["snapshot"], record["period"], record["settings"])
        if first is None:
            first = same
        elif same != first:
            raise ValueError(f"RUN {record['run_id']} DIFFERS IN SNAPSHOT, PERIOD OR SETTINGS from the first run")
        keep = sensors.get(str(Path(run_dir)))
        dropped = []
        if keep is not None:
            missing = sorted(set(keep) - set(record["counts"]))
            if missing:
                raise ValueError(f"RUN {record['run_id']} HAS NO SENSORS {missing}")
            dropped = sorted(set(record["counts"]) - set(keep))
            record["counts"] = {s: c for s, c in record["counts"].items() if s in keep}
        model = aliases.get(str(Path(run_dir)), record["model"])
        source = {"run_id": record["run_id"], "model": record["model"], "seed": record["seed"], "dir": str(run_dir),
                  "counts": record["counts"], "kept": keep is not None, "dropped": dropped}
        record.update(dir=str(run_dir), dirs=[str(run_dir)], sources={record["run_id"]: record["model"]},
                      runs=[source])
        if model in records:
            merged = records[model]
            if not all(d in aliases for d in [*merged["dirs"], str(Path(run_dir))]):
                raise ValueError(f"TWO RUNS OF MODEL {model}")
            flags = (_point_only(merged), _oracle(merged), merged.get("origin_stride", 1))
            if flags != (_point_only(record), _oracle(record), record.get("origin_stride", 1)):
                raise ValueError(f"RUNS MERGED AS {model} DIFFER IN point_only, ORACLE OR origin_stride")
            seeds = [r for r in merged["runs"] if r["model"] == record["model"]]
            if seeds:
                if record["seed"] in {r["seed"] for r in seeds}:
                    raise ValueError(f"RUNS MERGED AS {model}: SEED {record['seed']} OF {record['model']} TWICE")
                if set(seeds[0]["counts"]) != set(record["counts"]):
                    raise ValueError(f"RUNS MERGED AS {model}: SEEDS OF {record['model']} KEEP DIFFERENT SENSORS")
            else:
                shared = sorted(set(merged["counts"]) & set(record["counts"]))
                if shared:
                    raise ValueError(f"RUNS MERGED AS {model} SHARE SENSORS {shared}")
                merged["counts"] = {**merged["counts"], **record["counts"]}
            merged["dirs"].append(str(run_dir))
            merged["sources"][record["run_id"]] = record["model"]
            merged["runs"].append(source)
        else:
            records[model] = {**record, "model": model}
        frames.append(_read_metrics(source).assign(model=model, seed=record["seed"], config=record["model"]))
    for model, record in records.items():
        by_config = {}
        for source in record["runs"]:
            by_config.setdefault(source["model"], set()).add(source["seed"])
        if len({frozenset(seeds) for seeds in by_config.values()}) > 1:
            raise ValueError(f"CONFIGS MERGED AS {model} HAVE DIFFERENT SEEDS "
                             f"{ {config: sorted(seeds) for config, seeds in by_config.items()} }")
        record["seeds"] = sorted(next(iter(by_config.values())))
    return records, average_seeds(pd.concat(frames, ignore_index=True), records)


# One source run's metric rows, only for the sensors it keeps (load_runs: source["counts"]).
def _read_metrics(source: dict) -> pd.DataFrame:
    metrics = pd.read_csv(Path(source["dir"]) / "metrics.csv", parse_dates=["origin"])
    return metrics[metrics["sensor"].isin(source["counts"])]


# Per-origin mean over the seeds of each model with more than one seed: one row per (sensor, context, origin,
# horizon), every metric averaged. An origin that failed in any seed is "failed" with empty metrics (never imputed);
# else "fallback" if any seed fell back. Models with one seed keep their rows unchanged. The seeds of one config must
# have the same (sensor, context, origin, horizon) rows. Drops load_runs' "seed" and "config" columns.
def average_seeds(metrics: pd.DataFrame, records: dict[str, dict]) -> pd.DataFrame:
    keys = ["sensor", "context", "origin", "horizon_steps"]
    out = []
    for model, record in records.items():
        rows = metrics[metrics["model"] == model]
        if len(record["seeds"]) > 1:
            per_key = rows.groupby(["config", *keys]).size()
            if (per_key != len(record["seeds"])).any():
                raise ValueError(f"SEEDS MERGED AS {model} DIFFER IN ORIGINS")
            grouped = rows.groupby(keys, sort=False)
            rank = rows["status"].map({"ok": 0, "fallback": 1, "failed": 2})
            averaged = grouped[list(METRICS)].mean()
            averaged["status"] = rank.groupby([rows[k] for k in keys], sort=False).max().map(
                {0: "ok", 1: "fallback", 2: "failed"})
            averaged.loc[averaged["status"] == "failed", list(METRICS)] = np.nan
            averaged[["context_steps", "scale"]] = grouped[["context_steps", "scale"]].first()
            rows = averaged.reset_index().assign(model=model)
        out.append(rows[[c for c in metrics.columns if c not in ("seed", "config")]])
    return pd.concat(out, ignore_index=True)


# Aliases and kept sensors (load_runs) for runs reported as one model `name` from a validation picks file
# (validation_selection output: sensor, model, family, selected). Each run keeps the sensors whose selected pick is
# its config, e.g. a pooled run picked by only some sensors of its kind. Only the picks of the runs' model families
# count. A run whose config isn't a pick, and a pick without a run, are refused.
def picked_runs(picks_csv: Path, name: str, run_dirs: list[Path]) -> tuple[dict[str, str], dict[str, list[str]]]:
    configs = {str(Path(d)): json.loads((Path(d) / "run.json").read_text())["model"] for d in run_dirs}
    picks = pd.read_csv(picks_csv)
    picks = picks[picks["selected"] & picks["family"].isin({model_family(m) for m in configs.values()})]
    by_config = picks.groupby("model")["sensor"].apply(sorted).to_dict()
    for run_dir, config in configs.items():
        if config not in by_config:
            raise ValueError(f"RUN {run_dir} ({config}) IS NOT A PICK IN {picks_csv}")
    unrun = sorted(set(by_config) - set(configs.values()))
    if unrun:
        raise ValueError(f"PICKS WITHOUT A RUN: {unrun}")
    return {d: name for d in configs}, {d: by_config[config] for d, config in configs.items()}


# Keeps only the metric rows of the (sensor, origin) pairs that every model evaluated. Returns them and the pairs.
def common_origins(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs = metrics[["model", "sensor", "origin"]].drop_duplicates()
    shared = pairs.groupby(["sensor", "origin"])["model"].nunique()
    keep = shared[shared == pairs["model"].nunique()].index.to_frame(index=False)
    return metrics.merge(keep, on=["sensor", "origin"]), keep


# True if the run's model is scored as a point forecaster.
def _point_only(record: dict) -> bool:
    return bool(record.get("model_params", {}).get("point_only", False))


# Mean metrics per model, sensor, context and horizon over the origins that didn't fail, plus the origin counts and
# the number of seeds averaged per origin. Probabilistic metrics are NaN for point-only models.
def cells(metrics: pd.DataFrame, records: dict[str, dict]) -> pd.DataFrame:
    ok = metrics[metrics["status"] != "failed"]
    out = ok.groupby(KEYS)[list(METRICS)].mean().reindex(metrics.groupby(KEYS).size().index)
    out["n_origins"] = ok.groupby(KEYS).size()
    out["n_failed"] = metrics[metrics["status"] == "failed"].groupby(KEYS).size()
    out = out.fillna({"n_origins": 0, "n_failed": 0}).reset_index()
    point_only = out["model"].map(lambda m: _point_only(records[m]))
    out.loc[point_only, list(PROBABILISTIC)] = np.nan
    out["point_only"] = point_only
    out["n_seeds"] = out["model"].map(lambda m: len(records[m]["seeds"]))
    return out


# The spread over seeds: for each model with more than one seed, every seed's own cell means of MASE, SQL and COV80
# per sensor, context and horizon (origins that didn't fail; with common_origins only the kept pairs).
# Probabilistic metrics are NaN for point-only models.
def seed_cells(records: dict[str, dict], keep: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    columns = ["model", "seed", "sensor", "context", "horizon_steps", "MASE", "SQL", "COV80", "n_origins"]
    frames = []
    for model, record in records.items():
        if len(record["seeds"]) < 2:
            continue
        for source in record["runs"]:
            metrics = _read_metrics(source)
            if keep is not None:
                metrics = metrics.merge(keep, on=["sensor", "origin"])
            ok = metrics[metrics["status"] != "failed"].groupby(["sensor", "context", "horizon_steps"])
            table = ok[["MASE", "SQL", "COV80"]].mean().assign(n_origins=ok.size()).reset_index()
            if _point_only(record):
                table[["SQL", "COV80"]] = np.nan
            frames.append(table.assign(model=model, seed=source["seed"]))
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)[columns].sort_values(
        ["model", "sensor", "context", "horizon_steps", "seed"], ignore_index=True)


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
# The primary test uses statsmodels' lag rule max(DM_HORIZON - 1, ceil(n^(1/3))); "_28" is the same test with
# DM_LONG_LAGS lags (protocol P2).
def _dm_row(a: pd.Series, b: pd.Series, row: dict) -> dict:
    from statsmodels.tsa.api import diebold_mariano_test
    empty = {"statistic_28": np.nan, "pvalue_28": np.nan, "lags_28": np.nan}
    if set(a.index) != set(b.index) or a.isna().any() or b.isna().any():
        return {**row, "statistic": np.nan, "pvalue": np.nan, "lags": np.nan, "mean_loss_difference": np.nan,
                "note": "unpaired or failed origins", **empty}
    b = b.loc[a.index]
    if np.allclose(a.to_numpy(), b.to_numpy()):  # e.g. the same point forecast as the reference
        return {**row, "statistic": np.nan, "pvalue": np.nan, "lags": np.nan,
                "mean_loss_difference": float((a - b).mean()), "note": "identical losses", **empty}
    results = [diebold_mariano_test(np.zeros(len(a)), a.to_numpy(), b.to_numpy(), criterion=lambda y, f: f,
                                    horizon=DM_HORIZON, harvey_adj=True, lags=lags) for lags in (None, DM_LONG_LAGS)]
    primary, long = results
    return {**row, "statistic": float(primary.statistic), "pvalue": float(primary.pvalue), "lags": int(primary.lags),
            "mean_loss_difference": float((a - b).mean()), "note": "", "statistic_28": float(long.statistic),
            "pvalue_28": float(long.pvalue), "lags_28": int(long.lags)}


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
# - F3: chronos2 against the statistical baselines sf_mstl_arima and ag_savg672 (protocol P3; the best tuned trained
#   model is added with extra tests once chosen on validation);
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
        tests += [FamilyTest("F3", "chronos2", context, m, context) for m in ("sf_mstl_arima", "ag_savg672")
                  if m in models]
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


# Circular block bootstrap of a per-origin series over consecutive Berlin calendar days (protocol P2): the days from
# the first to the last origin's day form a circle, days without origins included as empty days (so a gap keeps its
# length). Every resample joins ceil(n / block_days) blocks of block_days consecutive days, cut to n days, and takes
# the mean over all origins of the drawn days. Returns the mean and its 95 % percentile interval.
def block_bootstrap(values: pd.Series, block_days: int = BLOCK_DAYS, samples: int = BOOTSTRAP_SAMPLES,
                    seed: int = BOOTSTRAP_SEED) -> tuple[float, float, float]:
    by_day = values.groupby(_berlin_days(values.index))
    days = pd.date_range(min(by_day.groups), max(by_day.groups), freq="D").date
    sums = by_day.sum().reindex(days, fill_value=0.0).to_numpy()
    counts = by_day.size().reindex(days, fill_value=0).to_numpy()
    n = len(days)
    starts = np.random.default_rng(seed).integers(0, n, size=(samples, math.ceil(n / block_days)))
    pick = ((starts[:, :, None] + np.arange(block_days)) % n).reshape(samples, -1)[:, :n]
    drawn = counts[pick].sum(axis=1)
    if (drawn == 0).any():
        raise ValueError("A BLOCK RESAMPLE DREW ONLY DAYS WITHOUT ORIGINS")
    low, high = np.quantile(sums[pick].sum(axis=1) / drawn, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


# The wording rule of protocol P2 for a family test: "robust" when significant after Holm with both the primary lags
# and DM_LONG_LAGS, "depends on the bandwidth" when with only one of them.
def bandwidth_wording(primary: bool, long: bool) -> str:
    if primary and long:
        return "robust"
    return "depends on the bandwidth" if primary or long else "not significant"


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
# - the mean MASE difference (model − against) with its 95 % intervals from the 7-day block bootstrap ("_block7",
#   the effect-size interval of protocol P2) and the single-day bootstrap, and the difference relative to the other
#   model's mean MASE;
# - Holm's correction of the MAE p-values within each family, for the primary lags and for DM_LONG_LAGS ("_28"),
#   and the bandwidth wording of the two decisions.
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
                _, low7, high7 = block_bootstrap(diff)
                row.update({"MASE_difference": mean, "MASE_difference_low": low, "MASE_difference_high": high,
                            "MASE_relative_difference": mean / float(b.mean()),
                            "interval_excludes_0": bool(low > 0 or high < 0),
                            "MASE_difference_low_block7": low7, "MASE_difference_high_block7": high7,
                            "interval_excludes_0_block7": bool(low7 > 0 or high7 < 0)})
            out.append(row)
    table = pd.DataFrame(out)
    if len(table):
        for suffix in ("", "_28"):
            table[f"pvalue_holm{suffix}"], table[f"significant_holm{suffix}"] = np.nan, False
            for _, rows in table.groupby("family"):
                adjusted, significant = holm(rows[f"pvalue{suffix}"])
                table.loc[rows.index, f"pvalue_holm{suffix}"] = adjusted
                table.loc[rows.index, f"significant_holm{suffix}"] = significant
        table["bandwidth"] = [bandwidth_wording(a, b) for a, b in zip(table["significant_holm"],
                                                                      table["significant_holm_28"])]
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


# The model family a run belongs to for selection: its name without a covariate set ("+<set>") and without a tuned
# config ("_tuned_<config>"), e.g. "ag_deepar_tuned_c672-lr0.001+cal" → "ag_deepar".
def model_family(model: str) -> str:
    return model.split("+")[0].split("_tuned_")[0]


# Selection on validation (Phase 4 rule E, Phase 6a): per sensor and model family, the candidate with the lowest mean
# SQL at 24 h over the sensor's valid validation origins at the given context.
# The candidates must be comparable: load_runs() refuses runs from different snapshots, periods or settings and two
# runs of one model; this refuses runs that aren't validation runs and a context no run has.
# Oracle runs, runs on uncommitted code (code.dirty) and candidates with a failed origin are never picked.
# Returns one row per (sensor, candidate).
def validation_selection(run_dirs: list[Path], context: str = "28d") -> pd.DataFrame:
    records, metrics = load_runs(run_dirs)
    for model, record in records.items():
        if record["period"] != "validation":
            raise ValueError(f"{record['dir']} IS NOT A VALIDATION RUN")
    metrics = metrics[(metrics["horizon_steps"] == max(REPORT_STEPS)) & (metrics["context"] == context)]
    if not len(metrics):
        raise ValueError(f"NO RUN HAS ROWS AT CONTEXT {context} (contexts are labelled like '28d')")
    rows = []
    for (model, sensor), cell in metrics.groupby(["model", "sensor"]):
        ok = cell[cell["status"] != "failed"]
        record = records[model]
        rows.append({"sensor": sensor, "model": model, "family": model_family(model), "run_id": record["run_id"],
                     "n_origins": len(ok), "failed": len(cell) - len(ok), "mean_SQL": ok["SQL"].mean(),
                     "mean_MASE": ok["MASE"].mean(), "oracle": _oracle(record),
                     "dirty": bool(record["code"].get("dirty"))})
    table = pd.DataFrame(rows)
    eligible = table[~table["oracle"] & ~table["dirty"] & (table["failed"] == 0) & table["mean_SQL"].notna()]
    picks = eligible.loc[eligible.groupby(["sensor", "family"])["mean_SQL"].idxmin()]
    table["selected"] = table.index.isin(picks.index)
    return table.sort_values(["sensor", "family", "mean_SQL"], ignore_index=True)


# Returns two tables:
# - the share of negative 0.1 and 0.5 quantiles per model, sensor, context and horizon;
# - the metrics of the visitor sensors with every quantile clipped at 0 (a sensitivity table only).
# Reads each source run's forecasts.csv.gz, only for the sensors it keeps. keep: only these (sensor, origin) pairs
# (None = all).
# A model with several seeds pools its seeds' forecast rows: the shares and COV80_open are means over seeds and
# origins, and the clipped scores are computed per seed and origin, then averaged.
# opened: open_slots() rows; with them, the negative table also gets COV80_open and n_open_steps for the halls
# (empty for point-only models and without opened).
def forecast_tables(records: dict[str, dict], metrics: pd.DataFrame, tasks: dict[str, SensorTask],
                    keep: Optional[pd.DataFrame] = None,
                    opened: Optional[pd.DataFrame] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    visitors = {name for name, task in tasks.items() if task.kind == "visitors"}
    negative_rows, clipped_rows = [], []
    for model, record in records.items():
        forecasts = []
        for source in record["runs"]:
            frame = pd.read_csv(Path(source["dir"]) / "forecasts.csv.gz", parse_dates=["origin"])
            forecasts.append(frame[frame["sensor"].isin(source["counts"])].assign(run=source["run_id"]))
        forecasts = pd.concat(forecasts, ignore_index=True)
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
        for (_, sensor, context, origin), window in forecasts[forecasts["sensor"].isin(visitors)].groupby(
                ["run", "sensor", "context", "origin"]):
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


# Origin counts per model, source run, sensor and context from each run.json (kept sensors only):
# valid and evaluated origins, ok, fallback, failed, and crossed quantile steps.
# With an origin stride fewer origins are evaluated. Runs from before the stride option evaluated every valid origin.
def failures(records: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for model, record in records.items():
        for source in record["runs"]:
            for sensor, counts in source["counts"].items():
                for context, c in counts["contexts"].items():
                    rows.append({"model": model, "sensor": sensor, "context": context,
                                 "valid_origins": counts["valid"],
                                 "evaluated_origins": counts.get("evaluated", counts["valid"]),
                                 **{k: c.get(k, 0) for k in ("ok", "fallback", "failed", "crossed_steps")},
                                 "run_id": source["run_id"], "seed": source["seed"]})
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


# The family tests (Holm within each family): DM on MAE at 24 h with both lag settings and the bandwidth wording, and
# the MASE difference with its block-bootstrap and day-bootstrap intervals.
def _family_table(family_table: pd.DataFrame) -> str:
    if not len(family_table):
        return "# Test families\n\nNo family test has both of its models in this report.\n"
    columns = ["family", "model", "context", "against", "against_context", "sensor", "n_origins",
               "mean_loss_difference", "pvalue", "pvalue_holm", "significant_holm", "pvalue_28", "pvalue_holm_28",
               "bandwidth", "MASE_difference", "MASE_difference_low_block7", "MASE_difference_high_block7",
               "MASE_difference_low", "MASE_difference_high", "note"]
    table = family_table[[c for c in columns if c in family_table]]
    return ("# Test families (frozen protocol F1-F7; Holm within each family, alpha 0.05)\n\n"
            "DM on per-origin MAE at 24 h with statsmodels' lag rule (pvalue) and with "
            f"{DM_LONG_LAGS} lags (pvalue_28, 7 days of origins); \"robust\" = significant after Holm under both "
            "(protocol P2). MASE difference = model − against, with 95 % intervals from a circular bootstrap over "
            f"blocks of {BLOCK_DAYS} Berlin calendar days (_block7, the effect-size interval) and over single Berlin "
            f"days ({BOOTSTRAP_SAMPLES} resamples each). Negative = the model is better.\n\n"
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
    seeded = sorted(f"{m} (seeds {', '.join(map(str, r['seeds']))})" for m, r in records.items()
                    if len(r["seeds"]) > 1)
    if seeded:
        parts.append(f"**Seed-averaged:** {'; '.join(seeded)}. Per-origin losses are averaged over the seeds before "
                     "every table and test; each seed's own cells are in seeds.csv.\n")
    filtered = sorted(m for m, r in records.items() if any(s["dropped"] for s in r["runs"]))
    if filtered:
        parts.append(f"**Kept sensors only:** {', '.join(filtered)} report only the sensors that picked each run "
                     "(report.json kept_sensors); the other sensors of those runs are in no table.\n")
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
# aliases: run folder → model name, to report several runs (configs or seeds) as one model; sensors: run folder →
# the sensors it keeps (load_runs, picked_runs).
# picks: model name → the validation picks file its aliases and sensors came from (recorded with its SHA-256).
# Returns the folder.
def make_report(run_dirs: list[Path], out_root: Path, reference: tuple[str, int],
                tasks: Optional[dict[str, SensorTask]] = None, report_id: Optional[str] = None,
                common: bool = False, families: Optional[list[FamilyTest]] = None,
                snapshot_dir: Optional[Path] = None, aliases: Optional[dict[str, str]] = None,
                sensors: Optional[dict[str, list[str]]] = None, picks: Optional[dict[str, Path]] = None) -> Path:
    tasks = TASKS if tasks is None else tasks
    started = datetime.now(timezone.utc)
    report_dir = Path(out_root) / (report_id or f"{started:%Y-%m-%dT%H%M%SZ}_report")
    if report_dir.exists():
        raise FileExistsError(f"{report_dir} EXISTS; reports are never overwritten")
    records, metrics = load_runs(run_dirs, aliases, sensors)
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
    seed_cells(records, keep).to_csv(report_dir / "seeds.csv", index=False)
    if family_table is not None:
        family_table.to_csv(report_dir / "families.csv", index=False)
    n_common = None if keep is None else len(keep)
    (report_dir / "summary.md").write_text(summary_markdown(cell_table, skill_table, records, reference, n_common,
                                                            tasks, family_table))
    first = next(iter(records.values()))
    (report_dir / "report.json").write_text(json.dumps({
        "report_id": report_dir.name, "runs": {m: r["run_id"] for m, r in records.items()},
        "merged": {m: r["sources"] for m, r in records.items() if len(r["sources"]) > 1},
        "seeds": {m: r["seeds"] for m, r in records.items() if len(r["seeds"]) > 1},
        "kept_sensors": {s["run_id"]: sorted(s["counts"]) for r in records.values() for s in r["runs"] if s["kept"]},
        "picks": {m: {"file": str(path), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
                  for m, path in (picks or {}).items()},
        "snapshot": first["snapshot"], "period": first["period"], "reference": list(reference),
        "dm": {"horizon": DM_HORIZON, "harvey_adj": True, "long_lags": DM_LONG_LAGS}, "common_origins": n_common,
        "families": None if families is None else [vars(test) for test in families],
        "holm_alpha": ALPHA, "bootstrap": {"samples": BOOTSTRAP_SAMPLES, "seed": BOOTSTRAP_SEED, "unit": "Berlin day"},
        "block_bootstrap": {"samples": BOOTSTRAP_SAMPLES, "seed": BOOTSTRAP_SEED, "block_days": BLOCK_DAYS,
                            "unit": "Berlin calendar day, circular"},
        "open_slots": None if snapshot_dir is None else {"context_days": OPEN_CONTEXT_DAYS,
                                                         "threshold": OPEN_THRESHOLD},
        "code": git_state(),
        "created_utc": started.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2))
    logger.info(f"REPORT WRITTEN to {report_dir}")
    return report_dir
