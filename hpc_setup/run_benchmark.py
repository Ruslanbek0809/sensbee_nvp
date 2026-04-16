#!/usr/bin/env python3

# Runs a systematic evaluation of the NVP forecasting pipeline across sensors, window sizes, providers/models, and parameter configurations.

# Outputs:
#   results/report_benchmark_<timestamp>.json   — raw data for every run
#   results/figures/                            — publication-quality PDF plots
#   results/tables/                             — LaTeX-ready metric tables

# Usage:
#   # Groq API — full window sweep (1-day, 7-day, 14-day):
#   export GROQ_API_KEY='...'
#   python hpc_setup/run_benchmark.py --provider groq
#
#   # Local LLM — specific model:
#   export HF_HUB_OFFLINE=1
#   python hpc_setup/run_benchmark.py --provider local --model llama2-7b
#
#   # Quick smoke test:
#   python hpc_setup/run_benchmark.py --provider groq --quick
#
#   # Ablation study (parameter sweep across all window sizes):
#   python hpc_setup/run_benchmark.py --provider groq --mode ablation

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ── Dataset registry ──────────────────────────────────────────────────────────

DATASETS = {
    "temperature": {
        "file": "temp_14day_sensbee_data.json",
        "column": "temperature",
        "unit": "°C",
        "description": "Outdoor temperature — Manebach weather station (15-min regular)",
        "horizon_hours": 24,
    },
    "visitors": {
        "file": "eishalle_14day_sensbee_data.json",
        "column": "visitors_total",
        "unit": "visitors",
        "description": "Visitor count — Ilmenau ice hall (MAX per 15-min, gap-filled)",
        "horizon_hours": 24,
    },
}

WINDOWS = {
    "1-day":  {"steps": 96,   "interval_min": 15},
    "7-day":  {"steps": 672,  "interval_min": 15},
    "14-day": {"steps": 1344, "interval_min": 15},
}

# Context length limits (in tokens) for different models.
# Used to truncate input when it would exceed the model's context.
MODEL_CONTEXT_LIMITS = {
    "gpt-3.5-turbo-instruct": 4097,  # OpenAI's limit
    "llama2-7b": 4096,               # Llama-2 base context
    "llama2-7b-chat": 4096,
    "llama2-13b": 4096,
    "mistral-7b": 8192,              # Mistral has 8K context
    "mistral-7b-instruct": 8192,
    # Groq Llama-3.3-70B has 128K context, no limit concerns
}

# Tokens per data point by provider/model type.
# Based on actual API error feedback:
# - GPT normalized (672 steps) = ~4102 prompt tokens → ~6.1 tokens/value
# - GPT raw values (494 steps) = ~3724 prompt tokens → ~7.5 tokens/value
# - Output max_tokens = horizon * 5 * 1.3 (from nvp_llms.py)
TOKENS_PER_VALUE_NORMALIZED = {
    "openai": 6.1,      # GPT with integer format + spacing
    "groq": 5.0,        # LLaMA-based
    "local": 5.0,       # LLaMA/Mistral local
    "mistral": 5.5,     # Mistral API
}

# Raw values use more tokens (decimal numbers like "23.456" vs integer "537")
TOKENS_PER_VALUE_RAW = {
    "openai": 7.8,      # GPT with decimal format (measured: 494 steps = 3724 tokens)
    "groq": 6.5,        # LLaMA-based with decimals
    "local": 6.5,       # LLaMA/Mistral local with decimals
    "mistral": 7.0,     # Mistral API with decimals
}

# Calculates the maximum number of input steps that fit within model's context. 
def get_max_input_steps(
    provider: str,
    model: Optional[str],
    horizon: int = 96,
    include_context: bool = False,
    use_normalization: bool = True,
) -> Optional[int]:
    # Determine model key for context limit
    model_key = None
    if provider == "openai":
        model_key = model or "gpt-3.5-turbo-instruct"
    elif provider == "local":
        model_key = model or "llama2-7b"
    
    if model_key not in MODEL_CONTEXT_LIMITS:
        return None  # No limit
    
    context_limit = MODEL_CONTEXT_LIMITS[model_key]
    
    # Select token rate based on normalization setting
    if use_normalization:
        tokens_per_value = TOKENS_PER_VALUE_NORMALIZED.get(provider, 5.0)
    else:
        tokens_per_value = TOKENS_PER_VALUE_RAW.get(provider, 7.0)
    
    # Output tokens calculated same as nvp_llms.py: horizon * 5 * 1.3
    output_tokens = int(horizon * 5 * 1.3)  # Match nvp_llms.py calculation
    
    # Prompt overhead (system message, formatting, etc.)
    prompt_overhead = 200
    if include_context:
        prompt_overhead += 100  # Statistical context adds tokens
    
    # Available tokens for input data
    available_for_prompt = context_limit - output_tokens
    available_for_data = available_for_prompt - prompt_overhead
    
    # Calculate max input steps with safety margin
    max_steps = int(available_for_data / tokens_per_value * 0.90)  # 10% safety margin
    
    return max(48, max_steps)  # At least 48 points (12 hours) minimum

# Best known config per LLMTime: normalize, no context, temp 0.9, 5 samples
BEST_PARAMS = {
    "num_forecasts": 5,
    "temperature": 0.9,
    "use_normalization": True,
    "include_context": False,
}

ABLATION_CONFIGS = [
    {"name": "LLMTime-like (baseline)",      "params": {**BEST_PARAMS}},
    {"name": "With statistical context",      "params": {**BEST_PARAMS, "include_context": True}},
    {"name": "Raw values — no normalisation", "params": {**BEST_PARAMS, "use_normalization": False}},
    {"name": "Raw values + context",          "params": {**BEST_PARAMS, "use_normalization": False, "include_context": True}},
    {"name": "Low temperature (0.7)",         "params": {**BEST_PARAMS, "temperature": 0.7}},
    {"name": "High temperature (1.0)",        "params": {**BEST_PARAMS, "temperature": 1.0}},
    {"name": "10 forecast samples",           "params": {**BEST_PARAMS, "num_forecasts": 10}},
    {"name": "3 forecast samples",            "params": {**BEST_PARAMS, "num_forecasts": 3}},
]


# ── Metrics ───────────────────────────────────────────────────────────────────

# Computes the Continuous Ranked Probability Score (CRPS) from forecast samples and ground truth.
def compute_crps(samples: List[np.ndarray], truth: np.ndarray) -> float:
    n_samples = len(samples)
    if n_samples == 0:
        return float("inf")

    sample_matrix = np.stack(samples, axis=0)  # (n_samples, horizon)

    abs_diff_truth = np.mean(np.abs(sample_matrix - truth[None, :]), axis=0)

    if n_samples > 1:
        pairwise = 0.0
        count = 0
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                pairwise += np.abs(sample_matrix[i] - sample_matrix[j])
                count += 1
        abs_diff_samples = pairwise / count if count > 0 else 0.0
    else:
        abs_diff_samples = 0.0

    crps_per_step = abs_diff_truth - 0.5 * abs_diff_samples
    return float(np.mean(crps_per_step))


# Computes core evaluation metrics: MAE, RMSE, sMAPE, WBA-10, naive MAE, CRPS.
def compute_metrics(
    forecast: np.ndarray,
    truth: np.ndarray,
    input_series: pd.Series,
    all_samples: Optional[List[np.ndarray]] = None,
) -> Dict[str, Any]:
    naive = np.full(len(truth), input_series.iloc[-1])

    mae  = float(np.mean(np.abs(forecast - truth)))
    rmse = float(np.sqrt(np.mean((forecast - truth) ** 2)))

    denom = np.abs(forecast) + np.abs(truth)
    with np.errstate(divide="ignore", invalid="ignore"):
        smape = float(
            np.mean(np.where(denom == 0, 0.0, 2 * np.abs(forecast - truth) / denom)) * 100
        )

    data_range = float(truth.max() - truth.min())
    if data_range > 0:
        wba10 = float(np.mean(np.abs(forecast - truth) <= 0.10 * data_range) * 100)
    else:
        wba10 = float(np.mean(forecast == truth) * 100)

    naive_mae    = float(np.mean(np.abs(naive - truth)))
    mae_vs_naive = round((1 - mae / naive_mae) * 100, 1) if naive_mae > 0 else None

    crps = None
    if all_samples and len(all_samples) > 1:
        crps = round(compute_crps(all_samples, truth), 4)

    return {
        "mae":          round(mae,  4),
        "rmse":         round(rmse, 4),
        "smape":        round(smape, 2),
        "wba_10":       round(wba10, 1),
        "naive_mae":    round(naive_mae, 4),
        "mae_vs_naive": mae_vs_naive,
        "crps":         crps,
    }


# Computes a quality score based on forecast characteristics and input data range.
def compute_quality_score(forecast: np.ndarray, input_series: pd.Series) -> Dict[str, Any]:
    vals = input_series.values
    in_min, in_max = float(vals.min()), float(vals.max())
    in_range = in_max - in_min

    f_std  = float(np.std(forecast))
    f_mean = float(np.mean(forecast))

    is_constant = bool(len(np.unique(np.round(forecast, 3))) <= 2)
    is_flat     = bool(f_std < 0.1)
    is_linear   = bool(len(forecast) > 4 and
                       np.all(np.abs(np.diff(forecast, 2)) < 0.01))
    buffer      = max(in_range * 2.0, 10.0)
    in_bounds   = bool(float(forecast.min()) >= in_min - buffer and
                       float(forecast.max()) <= in_max + buffer)

    score = 100
    if is_constant: score -= 50
    elif is_flat:   score -= 30
    if is_linear:   score -= 20
    if not in_bounds: score -= 30
    if f_std < 0.5: score -= 10

    return {
        "quality_score": max(0, score),
        "forecast_std":  round(f_std,  3),
        "forecast_mean": round(f_mean, 3),
        "is_flat":      is_flat,
        "is_constant":  is_constant,
        "is_linear":    is_linear,
        "in_bounds":    in_bounds,
    }


# ── Single test runner ────────────────────────────────────────────────────────

# Runs a single test for a given sensor, window, and parameter configuration.
def run_one_test(
    sensor_key: str,
    window_key: str,
    param_config: Dict[str, Any],
    provider: str,
    model: Optional[str],
    data_dir: Path,
) -> Dict[str, Any]:

    from src.data_access.data_loader import load_sensor_series_from_json, split_series_for_evaluation
    from src.models.nvp_llms import nvp_llms_forecast

    ds  = DATASETS[sensor_key]
    win = WINDOWS[window_key]
    horizon = max(1, (ds["horizon_hours"] * 60) // win["interval_min"])

    # Calculate max input steps based on model context limit
    include_context = param_config.get("include_context", False)
    use_normalization = param_config.get("use_normalization", True)
    max_steps = get_max_input_steps(
        provider=provider,
        model=model,
        horizon=horizon,
        include_context=include_context,
        use_normalization=use_normalization,
    )
    
    # Truncate requested steps if they exceed context limit
    requested_steps = win["steps"]
    actual_steps = requested_steps
    truncated = False
    if max_steps is not None and requested_steps > max_steps:
        actual_steps = max_steps
        truncated = True
        logger.warning(
            f"Requested {requested_steps} input steps but only {max_steps} fit in context. "
            f"Truncating input."
        )

    base = {
        "sensor":        sensor_key,
        "window":        window_key,
        "config_name":   param_config.get("_name", "default"),
        "horizon_hours": ds["horizon_hours"],
        "interval_min":  win["interval_min"],
        "truncated":     truncated,
        "requested_steps": requested_steps,
        "actual_steps":  actual_steps,
    }

    try:
        series = load_sensor_series_from_json(
            path=str(data_dir / ds["file"]),
            column_name=ds["column"],
            resample_interval_minutes=win["interval_min"],
        )
    except Exception as exc:
        return {**base, "error": f"Data load failed: {exc}"}

    try:
        input_series, ground_truth = split_series_for_evaluation(
            series, horizon_steps=horizon, input_steps=actual_steps
        )
    except ValueError as exc:
        return {**base, "error": str(exc)}

    if len(input_series) < 12:
        return {**base, "error": f"Too few input points: {len(input_series)}"}

    model_params = {k: v for k, v in param_config.items() if not k.startswith("_")}

    start = time.time()
    try:
        forecast = nvp_llms_forecast(
            series=input_series,
            horizon=horizon,
            provider=provider,
            model=model,
            column_name=ds["column"],
            **model_params,
        )
    except Exception as exc:
        return {**base, "error": str(exc), "duration_s": round(time.time() - start, 1)}

    duration = round(time.time() - start, 1)
    metrics  = compute_metrics(forecast, ground_truth.values, input_series)
    quality  = compute_quality_score(forecast, input_series)

    return {
        **base,
        "input_points":  len(input_series),
        "horizon_steps": horizon,
        "provider":      provider,
        "model":         model or "default",
        "params":        {k: v for k, v in param_config.items() if not k.startswith("_")},
        "duration_s":    duration,
        "forecast_range": [round(float(forecast.min()), 3), round(float(forecast.max()), 3)],
        "truth_range":    [round(float(ground_truth.min()), 3), round(float(ground_truth.max()), 3)],
        "forecast_values": [round(float(v), 4) for v in forecast],
        "truth_values":    [round(float(v), 4) for v in ground_truth.values],
        "metrics":  metrics,
        "quality":  quality,
        "timestamp": datetime.now().isoformat(),
    }


# ── Plot generation ───────────────────────────────────────────────────────────

# Generates publication-quality PDF plots from benchmark results.
def generate_plots(results: List[Dict], output_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except ImportError:
        print("matplotlib not installed — skipping plot generation.")
        return

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    ok = [r for r in results if "error" not in r]
    if not ok:
        return

    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "figure.dpi": 150,
    })

    # --- Figure 1: Forecast vs truth for each sensor's best result ---
    for sensor_key in DATASETS:
        sensor_results = [r for r in ok if r["sensor"] == sensor_key]
        if not sensor_results:
            continue
        best = min(sensor_results, key=lambda x: x["metrics"]["mae"])

        fig, ax = plt.subplots(figsize=(10, 4))
        steps = np.arange(len(best["truth_values"]))
        truth = np.array(best["truth_values"])
        fcast = np.array(best["forecast_values"])
        naive = np.full(len(truth), truth[0])

        ax.plot(steps, truth, color="black", linewidth=1.5, label="Ground truth")
        ax.plot(steps, fcast, color="royalblue", linewidth=1.5, label="LLM forecast")
        ax.plot(steps, naive, color="gray", linestyle="--", linewidth=1, label="Naive baseline")

        unit = DATASETS[sensor_key]["unit"]
        ax.set_xlabel("Step (15-min intervals)")
        ax.set_ylabel(f"{sensor_key.title()} ({unit})")
        ax.set_title(
            f"{sensor_key.title()} | {best['window']} input | "
            f"{best.get('provider','?')} | MAE = {best['metrics']['mae']:.2f} {unit}"
        )
        ax.legend(loc="best")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        fig.tight_layout()
        fig.savefig(fig_dir / f"forecast_{sensor_key}_best.pdf")
        plt.close(fig)
        print(f"  Saved: figures/forecast_{sensor_key}_best.pdf")

    # --- Figure 2: MAE vs window bar chart ---
    sensors_in_results = sorted({r["sensor"] for r in ok})
    providers_in_results = sorted({r.get("provider", "?") for r in ok})

    for sensor_key in sensors_in_results:
        sensor_ok = [r for r in ok if r["sensor"] == sensor_key]
        windows_present = sorted({r["window"] for r in sensor_ok},
                                  key=lambda w: WINDOWS.get(w, {}).get("steps", 0))

        if len(windows_present) < 2:
            continue

        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(windows_present))
        width = 0.35
        offset = 0

        for provider in providers_in_results:
            maes = []
            for w in windows_present:
                subset = [r for r in sensor_ok
                          if r["window"] == w and r.get("provider", "?") == provider]
                if subset:
                    maes.append(np.mean([r["metrics"]["mae"] for r in subset]))
                else:
                    maes.append(0)
            ax.bar(x + offset, maes, width, label=provider)
            offset += width

        ax.set_xlabel("Input window")
        ax.set_ylabel(f"MAE ({DATASETS[sensor_key]['unit']})")
        ax.set_title(f"{sensor_key.title()} — MAE by window and provider")
        ax.set_xticks(x + width * (len(providers_in_results) - 1) / 2)
        ax.set_xticklabels(windows_present)
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / f"mae_vs_window_{sensor_key}.pdf")
        plt.close(fig)
        print(f"  Saved: figures/mae_vs_window_{sensor_key}.pdf")

    # --- Figure 3: Ablation quality bar chart ---
    ablation = [r for r in ok if r.get("config_name", "") != "default"
                and r.get("config_name", "") != "best-params"]
    if len(ablation) >= 4:
        fig, ax = plt.subplots(figsize=(9, 4))
        configs = sorted({r["config_name"] for r in ablation})
        scores  = []
        colors  = []
        for cfg in configs:
            subset = [r for r in ablation if r["config_name"] == cfg]
            avg_q = np.mean([r["quality"]["quality_score"] for r in subset])
            scores.append(avg_q)
            is_norm = any(r["params"].get("use_normalization", True) for r in subset)
            colors.append("steelblue" if is_norm else "coral")

        y_pos = np.arange(len(configs))
        ax.barh(y_pos, scores, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([c[:35] for c in configs], fontsize=9)
        ax.set_xlabel("Quality Score (0–100)")
        ax.set_title("Parameter Ablation — Quality Score")
        ax.set_xlim(0, 105)

        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor="steelblue", label="Normalised"),
                           Patch(facecolor="coral",     label="Raw values")]
        ax.legend(handles=legend_elements, loc="lower right")
        fig.tight_layout()
        fig.savefig(fig_dir / "ablation_quality.pdf")
        plt.close(fig)
        print(f"  Saved: figures/ablation_quality.pdf")


# ── Main ──────────────────────────────────────────────────────────────────────
# ── Output helpers ────────────────────────────────────────────────────────────

# Custom JSON encoder for numpy types.
class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# Prints a summary of the benchmark results.
def print_summary(results: List[Dict]) -> None:
    ok     = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]

    print(f"\n{'='*90}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*90}")
    print(f"Total: {len(results)}  |  OK: {len(ok)}  |  Failed: {len(failed)}")

    if not ok:
        return

    # Group by (sensor, window, config_name) and show metrics
    groups: Dict[Tuple, List] = {}
    for r in ok:
        key = (r["sensor"], r["window"], r.get("config_name", "default"))
        groups.setdefault(key, []).append(r)

    print(f"\n{'Sensor':<12} {'Window':<8} {'Config':<28} "
          f"{'MAE':>8} {'RMSE':>8} {'sMAPE%':>8} {'WBA-10%':>8} {'vs.Naive%':>10} {'Time(s)':>8}")
    print("-" * 110)

    for key in sorted(groups.keys()):
        items = groups[key]
        r = items[0]  # Each combination should have one result now
        m = r["metrics"]
        vs_s = f"{m['mae_vs_naive']:+.1f}" if m.get("mae_vs_naive") is not None else "N/A"
        config_short = key[2][:26] + ".." if len(key[2]) > 28 else key[2]

        print(
            f"{key[0]:<12} {key[1]:<8} {config_short:<28} "
            f"{m['mae']:>8.3f} {m['rmse']:>8.3f} {m['smape']:>8.1f} "
            f"{m['wba_10']:>8.1f} {vs_s:>10} {r['duration_s']:>8.1f}"
        )

    print(f"{'='*110}")

    # Best per sensor
    print("\nBest result per sensor (lowest MAE):")
    for sensor in DATASETS:
        subset = [r for r in ok if r["sensor"] == sensor]
        if subset:
            best = min(subset, key=lambda x: x["metrics"]["mae"])
            m = best["metrics"]
            print(f"  {sensor}: {best['window']} | {best.get('config_name','?')} | "
                  f"MAE={m['mae']:.3f} | WBA-10={m['wba_10']:.1f}% | vs.Naive={m.get('mae_vs_naive','N/A')}")

    if failed:
        print(f"\nFailed ({len(failed)}):")
        for r in failed:
            print(f"  {r.get('sensor','?')} | {r.get('window','?')} | "
                  f"{r.get('config_name','?')} → {r['error'][:80]}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SensBee NVP — Comprehensive Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--provider", default="groq",
                        choices=["groq", "local", "mistral", "openai"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--mode", default="ablation",
                        choices=["window", "ablation", "full"],
                        help="window=vary windows only; ablation=vary params across windows; full=everything")
    parser.add_argument("--sensor", default="all",
                        choices=["all"] + list(DATASETS.keys()))
    parser.add_argument("--windows", nargs="+", default=["all"],
                        choices=list(WINDOWS.keys()) + ["all"])
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 1 window, 1 sensor")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-tables", action="store_true")
    args = parser.parse_args()

    if args.provider == "groq" and not os.getenv("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY not set.")
        sys.exit(1)
    if args.provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set.")
        sys.exit(1)

    sensors = list(DATASETS.keys()) if args.sensor == "all" else [args.sensor]
    windows = list(WINDOWS.keys()) if "all" in args.windows else list(dict.fromkeys(args.windows))

    if args.quick:
        sensors = ["temperature"]
        windows = ["1-day"]

    # Check which windows will be truncated due to context limits
    for w in windows:
        max_steps = get_max_input_steps(args.provider, args.model)
        if max_steps is not None and WINDOWS[w]["steps"] > max_steps:
            print(f"NOTE: {w} window ({WINDOWS[w]['steps']} steps) will be truncated to {max_steps} steps for this model")

    # Build test plan (no repeats — variance comes from multiple forecasts per call)
    plan: List[Tuple[str, str, Dict]] = []

    if args.mode == "window":
        # Simple window sweep with best params
        for sensor in sensors:
            for window in windows:
                plan.append((sensor, window, {**BEST_PARAMS, "_name": "best-params"}))

    elif args.mode == "ablation":
        # Ablation study: each config tested across all window sizes
        for sensor in sensors:
            for window in windows:
                for cfg in ABLATION_CONFIGS:
                    plan.append((sensor, window, {**cfg["params"], "_name": cfg["name"]}))

    else:  # full
        for sensor in sensors:
            for window in windows:
                for cfg in ABLATION_CONFIGS:
                    plan.append((sensor, window, {**cfg["params"], "_name": cfg["name"]}))

    print("=" * 70)
    print("SENSBEE NVP — COMPREHENSIVE BENCHMARK")
    print("=" * 70)
    print(f"Mode:     {args.mode}")
    print(f"Provider: {args.provider} / {args.model or 'default'}")
    print(f"Sensors:  {sensors}")
    print(f"Windows:  {windows}")
    print(f"Total:    {len(plan)} tests")
    print()

    data_dir = PROJECT_ROOT / "data"
    results: List[Dict] = []

    for i, (sensor, window, params) in enumerate(plan, 1):
        label = params.get("_name", "")
        print(f"[{i}/{len(plan)}] {sensor} | {window} | {label} ...", end=" ", flush=True)

        result = run_one_test(
            sensor_key=sensor, window_key=window, param_config=params,
            provider=args.provider, model=args.model, data_dir=data_dir,
        )
        results.append(result)

        if "error" in result:
            print(f"FAILED — {result['error'][:200]}")
        else:
            m = result["metrics"]
            q = result["quality"]
            vs = f"{m['mae_vs_naive']:+.1f}%" if m.get("mae_vs_naive") is not None else "N/A"
            pts = result.get("actual_steps", "?")
            trunc = " (truncated)" if result.get("truncated") else ""
            print(f"MAE={m['mae']:.3f}  WBA-10={m['wba_10']:.1f}%  vs.naive={vs}  "
                  f"Q={q['quality_score']}/100  {result['duration_s']}s  ({pts} pts{trunc})")

        if args.provider in ("groq", "mistral", "openai") and i < len(plan):
            time.sleep(0.8)

    print_summary(results)

    # Save raw results
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"report_benchmark_{ts}.json"
    with open(out_path, "w") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "mode": args.mode,
            "provider": args.provider,
            "model": args.model,
            "windows": windows,
            "sensors": sensors,
            "num_results": len(results),
            "successful": sum(1 for r in results if "error" not in r),
            "failed": sum(1 for r in results if "error" in r),
            "results": results,
        }, f, indent=2, cls=_NumpyEncoder)
    print(f"\nResults saved → {out_path}")

    if not args.no_plots:
        print("\nGenerating plots...")
        generate_plots(results, out_dir)


if __name__ == "__main__":
    main()
