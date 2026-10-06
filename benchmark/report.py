# Report over several harness runs (benchmark/runner.py output folders), built only from their raw files: per-cell
# means, relative errors and skill against the reference (Seasonal Naive m = 96 at the 28-day context, decision D-B6),
# Diebold–Mariano tests per sensor, failure counts, the share of negative quantiles, and a sensitivity table for the
# visitor counts with every model's quantiles clipped at 0 (D-B3: clipping is never part of the main tables).
#
# All runs must come from the same snapshot, period and harness settings. A model scored as a point forecaster
# (run.json model_params.point_only, e.g. Theta, D-B4) gets no probabilistic metrics. A failed origin is never imputed:
# its relative error and DM test are left empty with a note.

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from benchmark.metrics import METRICS, relative_errors, relative_skill, window_metrics
from benchmark.snapshot import git_state
from benchmark.tasks import QUANTILE_LEVELS, REPORT_STEPS, TASKS, SensorTask

logger = logging.getLogger(__name__)

REFERENCE_LABEL = "reference"
PROBABILISTIC = ("QL", "SQL", "COV80")
DM_HORIZON = 4  # 24-h windows every 6 h overlap 4 origins: HAC lags ≥ 3
QUANTILE_COLUMNS = [f"q{level:g}" for level in QUANTILE_LEVELS]
KEYS = ["model", "sensor", "context", "horizon_steps"]


# Reads each run's run.json and metrics.csv. Refuses runs that differ in snapshot, period or settings, and two runs of
# the same model. Returns (records by model, metrics of all runs with a "model" column).
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


# True if the run's model is scored as a point forecaster.
def _point_only(record: dict) -> bool:
    return bool(record.get("model_params", {}).get("point_only", False))


# Mean metrics per model, sensor, context and horizon over the origins that didn't fail, with the origin counts.
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
# (MASE, SQL). Origins are paired; a model with failed origins on a sensor gets no relative error and a note.
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


# Skill per model, context, horizon and metric: the clipped geometric mean of the relative errors over the main
# sensors (parking is secondary), only for cells that have a relative error on every main sensor.
def skill(relative_table: pd.DataFrame, tasks: dict[str, SensorTask]) -> pd.DataFrame:
    main = sorted(name for name, task in tasks.items() if not task.secondary)
    out = []
    for (metric, h, model, context), cell in relative_table.groupby(
            ["metric", "horizon_steps", "model", "context"], sort=False):
        cell = cell[cell["sensor"].isin(main)]
        complete = cell["relative_error"].notna().sum() == len(main)
        row = {"metric": metric, "horizon_steps": h, "model": model, "context": context, "n_sensors": len(main)}
        if complete:
            result = relative_skill(cell[["model", "relative_error"]]).iloc[0]
            out.append({**row, "gmean_relative_error": result["gmean_relative_error"], "skill": result["skill"],
                        "note": ""})
        else:
            out.append({**row, "gmean_relative_error": np.nan, "skill": np.nan,
                        "note": "not every main sensor has a paired relative error"})
    return pd.DataFrame(out)


# Diebold–Mariano test of a model against the reference on per-origin losses of one sensor (MAE, and QL for
# probabilistic models), with Newey–West lags from horizon=DM_HORIZON and the Harvey small-sample correction. Negative
# statistics mean the model's loss is lower. Needs statsmodels (bench venv).
def dm_tests(metrics: pd.DataFrame, records: dict[str, dict], reference: tuple[str, int]) -> pd.DataFrame:
    from statsmodels.tsa.api import diebold_mariano_test
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
                    row = {"loss": loss, "horizon_steps": h, "model": model, "context": context, "sensor": sensor,
                           "n_origins": len(a)}
                    if set(a.index) != set(b.index) or a.isna().any() or b.isna().any():
                        out.append({**row, "statistic": np.nan, "pvalue": np.nan, "lags": np.nan,
                                    "mean_loss_difference": np.nan, "note": "unpaired or failed origins"})
                        continue
                    b = b.loc[a.index]
                    if np.allclose(a.to_numpy(), b.to_numpy()):  # e.g. the same point forecast as the reference
                        out.append({**row, "statistic": np.nan, "pvalue": np.nan, "lags": np.nan,
                                    "mean_loss_difference": float((a - b).mean()), "note": "identical losses"})
                        continue
                    result =diebold_mariano_test(np.zeros(len(a)), a.to_numpy(), b.to_numpy(),
                                                  criterion=lambda y, f: f, horizon=DM_HORIZON, harvey_adj=True)
                    out.append({**row, "statistic": float(result.statistic), "pvalue": float(result.pvalue),
                                "lags": int(result.lags), "mean_loss_difference": float((a - b).mean()), "note": ""})
    return pd.DataFrame(out)


# Share of negative 0.1 and 0.5 quantiles per model, sensor, context and horizon, and the metrics of the visitor
# sensors with every quantile clipped at 0 (a sensitivity table only). Reads each run's forecasts.csv.gz.
def forecast_tables(records: dict[str, dict], metrics: pd.DataFrame,
                    tasks: dict[str, SensorTask]) -> tuple[pd.DataFrame, pd.DataFrame]:
    visitors = {name for name, task in tasks.items() if task.kind == "visitors"}
    negative_rows, clipped_rows = [], []
    for model, record in records.items():
        forecasts = pd.read_csv(Path(record["dir"]) / "forecasts.csv.gz", parse_dates=["origin"])
        forecasts = forecasts[forecasts["status"] != "failed"]
        for h in REPORT_STEPS:
            head = forecasts[forecasts["step"] <= h]
            grouped = head.groupby(["sensor", "context"])
            shares = pd.DataFrame({"negative_q0.1": grouped["q0.1"].apply(lambda q: float((q < 0).mean())),
                                   "negative_q0.5": grouped["q0.5"].apply(lambda q: float((q < 0).mean()))})
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
    negative = pd.concat(negative_rows, ignore_index=True)[KEYS + ["negative_q0.1", "negative_q0.5"]]
    clipped = pd.DataFrame(clipped_rows)
    if len(clipped):
        clipped = clipped.groupby(KEYS)[list(METRICS)].mean().reset_index()
        point_only = clipped["model"].map(lambda m: _point_only(records[m]))
        clipped.loc[point_only, list(PROBABILISTIC)] = np.nan
    return negative, clipped


# Origin counts per model, sensor and context from each run.json: ok, fallback, failed and crossed quantile steps.
def failures(records: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for model, record in records.items():
        for sensor, counts in record["counts"].items():
            for context, c in counts["contexts"].items():
                rows.append({"model": model, "sensor": sensor, "context": context, "valid_origins": counts["valid"],
                             **{k: c.get(k, 0) for k in ("ok", "fallback", "failed", "crossed_steps")}})
    return pd.DataFrame(rows)


# A Markdown table from a DataFrame (no extra dependency); floats with 3 decimals, NaN as "n/a".
def _markdown(df: pd.DataFrame) -> str:
    def fmt(v) -> str:
        if isinstance(v, float):
            return "n/a" if math.isnan(v) else f"{v:.3f}"
        return str(v)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


# Sort key: context labels by their number of days ("1d" < "7d" < "14d"; "all" last), other columns as they are.
def _context_order(column: pd.Series) -> pd.Series:
    if column.name != "context":
        return column
    return column.map(lambda c: float("inf") if c == "all" else int(c.rstrip("d")))


# Headline tables: MASE and SQL per sensor and model at 24 h for each context, and the skill per model and context.
def summary_markdown(cell_table: pd.DataFrame, skill_table: pd.DataFrame, records: dict[str, dict],
                     reference: tuple[str, int]) -> str:
    first = next(iter(records.values()))
    parts = [f"# Baseline report\n\nSnapshot `{first['snapshot']['id']}`, period `{first['period']}`, "
             f"reference `{reference[0]}` at {reference[1]} days. Generated by `benchmark/report.py`; "
             "point-only models show n/a for probabilistic metrics. Parking is secondary (not in the skill).\n"]
    h = max(REPORT_STEPS)
    for metric in ("MASE", "SQL"):
        at_h = cell_table[cell_table["horizon_steps"] == h]
        table = at_h.pivot(index=["model", "context"], columns="sensor", values=metric)  # only cells that ran
        table["n_origins_total"] = at_h.groupby(["model", "context"])["n_origins"].sum().astype(int)
        table = table.reset_index()
        table = table.sort_values(["model", "context"], key=_context_order, ignore_index=True)
        parts.append(f"## {metric} at {h} steps (24 h), mean over origins\n\n{_markdown(table)}\n")
    for metric in ("MASE", "SQL"):
        table = skill_table[(skill_table["metric"] == metric) & (skill_table["horizon_steps"] == h)].sort_values(
            ["model", "context"], key=_context_order)
        parts.append(f"## Skill on {metric} at 24 h (1 − gmean of relative errors over the main sensors)\n\n"
                     f"{_markdown(table[['model', 'context', 'gmean_relative_error', 'skill', 'note']])}\n")
    return "\n".join(parts)


# Builds every table from the runs and writes them to out_root/<UTC>_report (refused if it exists). tasks: the sensor
# tasks the runs used (default: all TASKS). Returns the folder.
def make_report(run_dirs: list[Path], out_root: Path, reference: tuple[str, int],
                tasks: Optional[dict[str, SensorTask]] = None, report_id: Optional[str] = None) -> Path:
    tasks = TASKS if tasks is None else tasks
    started = datetime.now(timezone.utc)
    report_dir = Path(out_root) / (report_id or f"{started:%Y-%m-%dT%H%M%SZ}_report")
    if report_dir.exists():
        raise FileExistsError(f"{report_dir} EXISTS; reports are never overwritten")
    records, metrics = load_runs(run_dirs)
    cell_table = cells(metrics, records)
    relative_table = relative(metrics, records, reference)
    skill_table = skill(relative_table, tasks)
    dm_table = dm_tests(metrics, records, reference)
    negative, clipped = forecast_tables(records, metrics, tasks)
    cell_table = cell_table.merge(negative, on=KEYS, how="left")
    report_dir.mkdir(parents=True)
    cell_table.to_csv(report_dir / "cells.csv", index=False)
    relative_table.to_csv(report_dir / "relative.csv", index=False)
    skill_table.to_csv(report_dir / "skill.csv", index=False)
    dm_table.to_csv(report_dir / "dm.csv", index=False)
    clipped.to_csv(report_dir / "clipped_visitors.csv", index=False)
    failures(records).to_csv(report_dir / "failures.csv", index=False)
    (report_dir / "summary.md").write_text(summary_markdown(cell_table, skill_table, records, reference))
    first = next(iter(records.values()))
    (report_dir / "report.json").write_text(json.dumps({
        "report_id": report_dir.name, "runs": {m: r["run_id"] for m, r in records.items()},
        "snapshot": first["snapshot"], "period": first["period"], "reference": list(reference),
        "dm": {"horizon": DM_HORIZON, "harvey_adj": True}, "code": git_state(),
        "created_utc": started.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2))
    logger.info(f"REPORT WRITTEN to {report_dir}")
    return report_dir
