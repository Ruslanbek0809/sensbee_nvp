# Baseline forecasters for the benchmark harness (decisions D-B1–D-B6, docs in the thesis plan for the baselines
# step): AutoGluon's local models and statsforecast's MSTL behind the runner contract. Each entry gives a builder
# (returns the forecaster and the parameters recorded in run.json), the context lengths in days it may run on, and
# whether it is scored as a point forecaster. Heavy libraries load only when a forecaster is built or called, so this
# module imports in the service venv too.

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Callable

from benchmark.models.autogluon_models import autogluon_forecaster
from benchmark.models.statsforecast_models import mstl_forecaster
from benchmark.tasks import SEASON

WEEK = 7 * SEASON
REFERENCE = ("ag_snaive96", 28)  # model and context (days) every relative score is computed against (D-B6)
PACKAGES = ("autogluon.timeseries", "statsforecast", "statsmodels", "scipy", "torch", "coreforecast", "joblib")


# One baseline: build() → (forecaster, parameters); contexts: the allowed context lengths in days.
@dataclass(frozen=True)
class Baseline:
    build: Callable
    contexts: tuple[int, ...]


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


# Versions of the libraries behind the baselines, for run.json.
def library_versions() -> dict:
    out = {}
    for package in PACKAGES:
        try:
            out[package] = version(package)
        except PackageNotFoundError:
            out[package] = None
    return out
