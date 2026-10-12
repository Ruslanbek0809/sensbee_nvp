# Shared helpers for the benchmark tests, as fixtures: synthetic daily-periodic contexts, and the deciles of
# AutoGluon's own wrapper for a model (the cross-check for the zero-shot adapters; needs the bench venv).

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmark.tasks import HORIZON, QUANTILE_LEVELS

PATTERN = 10 + 5 * np.sin(np.arange(96) * 2 * np.pi / 96)


# n daily-periodic contexts of `days` days with a little noise, each with its own noise and end time.
def _contexts(n: int, days: int, noise: float = 0.3) -> list[pd.Series]:
    out = []
    for i in range(n):
        rng = np.random.default_rng(i)
        index = pd.date_range("2026-01-01", periods=days * 96, freq="15min") + pd.Timedelta(hours=6 * i)
        values = np.tile(np.roll(PATTERN, -24 * i), days) + rng.normal(0, noise, days * 96)
        out.append(pd.Series(values, index=index))
    return out


# Deciles of AutoGluon's own wrapper for one model on the given contexts (no fitting: a zero-shot model only loads).
def _autogluon_deciles(model: str, hyperparameters: dict, contexts: list[pd.Series], path: Path) -> np.ndarray:
    pytest.importorskip("autogluon.timeseries")
    from autogluon.timeseries import TimeSeriesPredictor

    from benchmark.models.autogluon_models import _batch_frame
    from benchmark.snapshot import BUCKET_FREQ
    data = _batch_frame(contexts)
    predictor = TimeSeriesPredictor(prediction_length=HORIZON, freq=BUCKET_FREQ, quantile_levels=list(QUANTILE_LEVELS),
                                    path=str(path), verbosity=0, log_to_file=False)
    predictor.fit(data, hyperparameters={model: hyperparameters}, enable_ensemble=False, skip_model_selection=True)
    predictions = predictor.predict(data)
    return np.stack([predictions.loc[f"{i:06d}", [str(q) for q in QUANTILE_LEVELS]].to_numpy(dtype=float)
                     for i in range(len(contexts))])


# make_contexts(n, days): n synthetic contexts (see _contexts).
@pytest.fixture
def make_contexts():
    return _contexts


# autogluon_deciles(model, hyperparameters, contexts, path): AutoGluon's deciles (see _autogluon_deciles).
@pytest.fixture
def autogluon_deciles():
    return _autogluon_deciles
