# Baseline forecasters for the benchmark harness (decisions D-B1–D-B6, docs in the thesis plan for the baselines
# step): AutoGluon's local models and statsforecast's MSTL behind the runner contract.
# Each entry gives:
# - a builder, which returns the forecaster and the parameters recorded in run.json;
# - the context lengths in days it may run on;
# - whether it is scored as a point forecaster.
# Heavy libraries load only when a forecaster is built or called, so this module imports in the service venv too.
# Zero-shot foundation models (ZERO_SHOT) are kept apart from the baselines.
# They need downloaded weights, so their tests run only with -m integration.
# Each is pinned to a Hugging Face model id and revision.
# Each runs on every context length its model can read: 1 to 28 days for all of them
# (2 days checks whether two daily cycles help over one), and longer ones only where the model's context limit allows.
# Trained models (TRAINED) build a trainer instead of a forecaster.
# The runner fits one model per sensor on the data before the period (benchmark/runner.py header).

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Callable, Optional

from benchmark.models.autogluon_models import autogluon_forecaster
from benchmark.models.chronos_models import chronos2_forecaster, chronos_bolt_forecaster
from benchmark.models.finetune_models import chronos2_finetune_trainer, finetune_grid
from benchmark.models.llmtime_models import llmtime_forecaster
from benchmark.models.mask_models import mask_forecaster, recent_mask_forecaster
from benchmark.models.moirai_models import moirai2_forecaster
from benchmark.models.statsforecast_models import mstl_forecaster
from benchmark.models.timesfm_models import timesfm25_forecaster
from benchmark.models.tirex_models import tirex_forecaster
from benchmark.models.toto_models import toto_forecaster
from benchmark.models.trained_models import (
    DEEPAR, DEEPAR_HOLDOUT, DIRECT_TABULAR, PATCHTST, RECURSIVE_TABULAR, TUNING_DAYS, autogluon_trainer, input_steps,
    tuned_grid,
)
from benchmark.covariates import KINDS as COVARIATE_KINDS
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
MISTRAL_7B = ("mistralai/Mistral-7B-v0.1", "27d67f1b5f57dc0953326b2601d68371d40ea8da")
# Gated (Llama 2 Community License, accepted on Hugging Face first); revision read from the HF API 2026-10-26. Its
# context is 4k tokens (model card), so 14 days don't fit (Phase 6a plan, 6a.6).
LLAMA2_7B = ("meta-llama/Llama-2-7b-hf", "01c7f73d771dfac7d292323805ebc428287df4f9")
PACKAGES = ("autogluon.timeseries", "statsforecast", "statsmodels", "scipy", "torch", "coreforecast", "joblib",
            "chronos-forecasting", "transformers", "huggingface_hub", "accelerate", "timesfm", "tirex-ts", "uni2ts",
            "lightgbm", "mlforecast", "gluonts", "lightning", "peft")


# One baseline.
# - build(): returns (forecaster, parameters), or (trainer, parameters) when trained.
# - contexts: the allowed context lengths in days.
# - weights: for a zero-shot model, (model id, revision, weight file name) in the Hugging Face cache.
# - trained: build() returns a trainer, which the runner fits per sensor.
# - covariate_kinds: the covariate kinds (benchmark/covariates.py) its forecaster or trainer accepts, none by default.
# - covariate_groups: the feature groups a covariate set must contain for it
#   (the "0 when closed" models need the opening profile).
@dataclass(frozen=True)
class Baseline:
    build: Callable
    contexts: tuple[int, ...]
    weights: Optional[tuple[str, str, str]] = None
    trained: bool = False
    covariate_kinds: tuple[str, ...] = ()
    covariate_groups: tuple[str, ...] = ()


# AutoGluon SeasonalNaive uses its seasonal branch only when the context has more than m + 1 steps.
# Otherwise it silently returns Naive (abstract_local_model.py:239).
# So m = 96 runs from 7 days and m = 672 runs from 14 days.
# Seasonal AutoARIMA (season 96) is not in the list. Even on 7-day contexts it took > 5 CPU-hours for 18% of the test
# workload (validation smoke, 2026-10-05), so the owner chose STL + ARIMA instead (sf_mstl_arima).
# MSTL needs two cycles of each season.
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
# A zero-shot model pinned to one (model id, revision).
# Its forecaster factory is built with that pin, and the cache check in its tests looks for the weight file at the
# same pin.
def _zero_shot(factory: Callable, pin: tuple[str, str], contexts: tuple[int, ...],
               weight_file: str = "model.safetensors", covariate_kinds: tuple[str, ...] = ()) -> Baseline:
    return Baseline(lambda: factory(*pin), contexts, (*pin, weight_file), covariate_kinds=covariate_kinds)


ZERO_SHOT: dict[str, Baseline] = {
    "chronos2": _zero_shot(chronos2_forecaster, CHRONOS2, (1, 2, 7, 14, 28, 56, 85), covariate_kinds=COVARIATE_KINDS),
    # Chronos-Bolt reads at most 2,048 steps, so the model cuts its 28-day context to the last 21.3 days.
    "chronos_bolt_base": _zero_shot(chronos_bolt_forecaster, CHRONOS_BOLT_BASE, (1, 2, 7, 14, 28)),
    # TimesFM 2.5 reads up to 16,256 steps next to its 128-step output block, so 169 days (16,224 steps) is its longest.
    "timesfm25": _zero_shot(timesfm25_forecaster, TIMESFM25, (1, 2, 7, 14, 28, 56, 85, 169)),
    # TiRex reads at most 2,048 steps, so the model cuts its 28-day context to the last 21.3 days.
    "tirex": _zero_shot(tirex_forecaster, TIREX, (1, 2, 7, 14, 28), "model.ckpt"),
    # Toto reads the last 4,096 steps (AutoGluon's default), more than the 2,688 of 28 days; it samples 256 paths.
    "toto": _zero_shot(toto_forecaster, TOTO, (1, 2, 7, 14, 28)),
    # Moirai-2 reads up to 512 tokens of 16 steps, more than 28 days need; it runs only in venv-moirai (uni2ts).
    "moirai2_small": _zero_shot(moirai2_forecaster, MOIRAI2_SMALL, (1, 2, 7, 14, 28)),
}


# Trained models (Phase 3): one model per sensor, fitted on the data before the period and then frozen.
# Their input window (at most 863 steps: DeepAR's 192 unrolled steps plus its 672-step lag) fits the 14-day context,
# so they run on 14 and 28 days.
# RecursiveTabular is a point forecaster (its intervals come from the context, not the model; decision D3-5).
# DeepAR and PatchTST stop early, using the final TUNING_DAYS before the cutoff.
TRAINED: dict[str, Baseline] = {
    "ag_direct_tabular": Baseline(lambda: autogluon_trainer("DirectTabular", DIRECT_TABULAR), (14, 28), trained=True,
                                  covariate_kinds=("known",)),
    "ag_recursive_tabular": Baseline(lambda: autogluon_trainer("RecursiveTabular", RECURSIVE_TABULAR,
                                                               point_only=True), (14, 28), trained=True,
                                     covariate_kinds=("known",)),
    "ag_deepar": Baseline(lambda: autogluon_trainer("DeepAR", DEEPAR, tuning_days=TUNING_DAYS,
                                                    holdout=DEEPAR_HOLDOUT), (14, 28), trained=True),
    "ag_patchtst": Baseline(lambda: autogluon_trainer("PatchTST", PATCHTST, tuning_days=TUNING_DAYS), (14, 28),
                            trained=True),
}


# Tuned trained models (Phase 6a, D-E5): one entry per grid point of trained_models.tuned_grid(), named
# "<model>_tuned_<config>". Contexts: 14 and 28 days when the input window fits 14 days, else 28 only.
# The tabular models and DeepAR accept known covariates (calendar inputs); the pick per sensor is made on validation.
def _tuned(base: str, model: str, params: dict) -> Baseline:
    contexts = (14, 28) if input_steps(params) <= 14 * SEASON else (28,)
    point_only = base == "ag_recursive_tabular"
    tuning = {"tuning_days": TUNING_DAYS} if base in ("ag_deepar", "ag_patchtst") else {}
    if base == "ag_deepar":
        tuning["holdout"] = params["context_length"] - 1
    kinds = () if base == "ag_patchtst" else ("known",)
    return Baseline(lambda: autogluon_trainer(model, params, point_only=point_only, **tuning), contexts, trained=True,
                    covariate_kinds=kinds)


TUNED: dict[str, Baseline] = {name: _tuned(*spec) for name, spec in tuned_grid().items()}


# Fine-tuned Chronos-2 (Phase 6a, 6a.7, RQ4): one entry per config of finetune_models.finetune_grid(), on 28-day
# contexts, values only; with --pooled, one fit per sensor kind. The pick per sensor is made on validation.
# chronos2_ft_lr0 is gate Z only (learning rate 0: must forecast like zero-shot Chronos-2); without "_tuned_" in its
# name it is never a candidate of the "chronos2_ft" family.
def _finetuned(config: dict) -> Baseline:
    return Baseline(lambda: chronos2_finetune_trainer(*CHRONOS2, **config), (28,), (*CHRONOS2, "model.safetensors"),
                    trained=True)


FINETUNED: dict[str, Baseline] = {
    **{name: _finetuned(config) for name, config in finetune_grid().items()},
    "chronos2_ft_lr0": _finetuned({"mode": "full", "learning_rate": 0.0, "num_steps": 10}),
}


# LLMTime (Phase 3, the research project's method) with Mistral-7B-v0.1, run locally.
# It uses the "LLMTime-like" setting of the research project (5 samples, temperature 0.9, normalized).
# It runs on 14-day contexts only (≈ 6-7k prompt tokens). The runs use a fixed origin subset (--origin-stride).
# It is kept apart from ZERO_SHOT so that the integration tests of ZERO_SHOT don't load 14.5 GB.
LLMTIME: dict[str, Baseline] = {
    "llmtime_mistral7b": _zero_shot(llmtime_forecaster, MISTRAL_7B, (14,), "model-00001-of-00002.safetensors"),
}


# "0 when closed" baselines (Phase 4, RQ2): a values-only base with every quantile set to 0 where the learned opening
# profile is below a threshold. build(threshold) → (forecaster, parameters).
# They run only with a covariate set that has "open_share" (cal_open), on the halls, at the base's contexts
# (Chronos-2 at 28 days only, the headline context).
# mask0_chronos2_recent (Phase 5, D5-2) takes the profile from each 28-day context instead, which is the rule the
# service applies. It needs no covariate set.
MASKED: dict[str, Baseline] = {
    "mask0_snaive672": Baseline(lambda threshold: mask_forecaster(BASELINES["ag_snaive672"].build, threshold,
                                                                  "ag_snaive672"),
                                (14, 28), covariate_kinds=("known",), covariate_groups=("open",)),
    "mask0_chronos2": Baseline(lambda threshold: mask_forecaster(ZERO_SHOT["chronos2"].build, threshold, "chronos2"),
                               (28,), ZERO_SHOT["chronos2"].weights, covariate_kinds=("known",),
                               covariate_groups=("open",)),
    "mask0_chronos2_recent": Baseline(lambda threshold: recent_mask_forecaster(ZERO_SHOT["chronos2"].build, threshold,
                                                                               "chronos2"),
                                      (28,), ZERO_SHOT["chronos2"].weights),
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
