# Baseline forecasters for the benchmark harness (decisions D-B1–D-B6, docs in the thesis plan for the baselines
# step): AutoGluon's local models and statsforecast's MSTL behind the runner contract. Each entry gives a builder
# (returns the forecaster and the parameters recorded in run.json), the context lengths in days it may run on, and
# whether it is scored as a point forecaster. Heavy libraries load only when a forecaster is built or called, so this
# module imports in the service venv too. Zero-shot foundation models (ZERO_SHOT) are kept apart from the baselines:
# they need downloaded weights, so their tests run only with -m integration. Each is pinned to a Hugging Face model id
# and revision, and runs on every context length its model can read: 1 to 28 days for all of them (2 days checks
# whether two daily cycles help over one), longer ones only for the models whose context limit allows them.

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Callable, Optional

from benchmark.models.autogluon_models import autogluon_forecaster
from benchmark.models.chronos_models import chronos2_forecaster, chronos_bolt_forecaster
from benchmark.models.statsforecast_models import mstl_forecaster
from benchmark.tasks import SEASON

WEEK = 7 * SEASON
REFERENCE = ("ag_snaive96", 28)  # model and context (days) every relative score is computed against (D-B6)
# Hugging Face model id and revision of each zero-shot model.
CHRONOS2 = ("amazon/chronos-2", "29ec3766d36d6f73f0696f85560a422f50e8498c")
CHRONOS_BOLT_BASE = ("amazon/chronos-bolt-base", "5d9f166d69f47aef3401367a7b842e78fe97b121")
TIMESFM25 = ("google/timesfm-2.5-200m-pytorch", "1d952420fba87f3c6dee4f240de0f1a0fbc790e3")
TIREX = ("NX-AI/TiRex", "63c740922493f5fbe60b277609ec62babfba2762")
TOTO = ("Datadog/Toto-Open-Base-1.0", "0411ceb27bdf7fc3e4892e99edc8ad08192dc3c5")
MOIRAI2_SMALL = ("Salesforce/moirai-2.0-R-small", "30f43ff08c8494f4943ae1521e9d4e94a0fbb389")
PACKAGES = ("autogluon.timeseries", "statsforecast", "statsmodels", "scipy", "torch", "coreforecast", "joblib",
            "chronos-forecasting", "transformers", "huggingface_hub", "accelerate", "timesfm", "tirex-ts", "uni2ts")


# One baseline: build() → (forecaster, parameters); contexts: the allowed context lengths in days; weights: for a
# zero-shot model, (model id, revision, weight file name) in the Hugging Face cache.
@dataclass(frozen=True)
class Baseline:
    build: Callable
    contexts: tuple[int, ...]
    weights: Optional[tuple[str, str, str]] = None


# AutoGluon SeasonalNaive uses its seasonal branch only when the context has more than m + 1 steps, otherwise it
# silently returns Naive (abstract_local_model.py:239), so m = 96 runs from 7 days and m = 672 from 14 days.
# Seasonal AutoARIMA (season 96) is not in the list: even on 7-day contexts it took > 5 CPU-hours for 18% of the test
# workload (validation smoke, 2026-10-05), so the owner chose STL + ARIMA instead (sf_mstl_arima). MSTL needs two
# cycles of each season.
BASELINES: dict[str, Baseline] = {
    "ag_naive": Baseline(lambda: autogluon_forecaster("Naive", {}), (1, 7, 14, 28)),
    "ag_snaive96": Baseline(lambda: autogluon_forecaster("SeasonalNaive", {"seasonal_period": SEASON}), (7, 14, 28)),
    "ag_snaive672": Baseline(lambda: autogluon_forecaster("SeasonalNaive", {"seasonal_period": WEEK}), (14, 28)),
    "ag_ets": Baseline(lambda: autogluon_forecaster(
        "AutoETS", {"seasonal_period": SEASON, "model": "ZZN", "damped": False}), (1, 7, 14, 28)),
    "ag_theta": Baseline(lambda: autogluon_forecaster(
        "Theta", {"seasonal_period": SEASON, "decomposition_type": "multiplicative"}, point_only=True), (1, 7, 14, 28)),
    "sf_mstl_d": Baseline(lambda: mstl_forecaster([SEASON]), (7, 14, 28)),
    "sf_mstl_dw": Baseline(lambda: mstl_forecaster([SEASON, WEEK]), (14, 28)),
    "sf_mstl_arima": Baseline(lambda: mstl_forecaster([SEASON], trend="arima"), (7, 14, 28)),
}

# Zero-shot foundation models. Chronos-2 reads up to 8,192 steps, so 85 days (8,160 steps) is its longest context.
ZERO_SHOT: dict[str, Baseline] = {
    "chronos2": Baseline(lambda: chronos2_forecaster(*CHRONOS2), (1, 2, 7, 14, 28, 56, 85),
                         (*CHRONOS2, "model.safetensors")),
    # Chronos-Bolt reads at most 2,048 steps: its 28-day context is cut to the last 21.3 days by the model.
    "chronos_bolt_base": Baseline(lambda: chronos_bolt_forecaster(*CHRONOS_BOLT_BASE), (1, 2, 7, 14, 28),
                                  (*CHRONOS_BOLT_BASE, "model.safetensors")),
}


# Versions of the libraries behind the baselines, for run.json.
def library_versions() -> dict:
    out = {}
    for package in PACKAGES:
        try:
            out[package] = version(package)
        except PackageNotFoundError:
            out[package] = None
    return out
