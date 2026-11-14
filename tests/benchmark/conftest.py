# Shared helpers for the benchmark tests, as fixtures:
# - synthetic daily-periodic contexts
# - the deciles of AutoGluon's own wrapper for a model (the cross-check for the zero-shot adapters, bench venv only)
# - the zero-shot forecasters, built once per test session
# - two synthetic halls over two seasons, shaped like the October snapshot (the January hall rules, P1)

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from benchmark.tasks import HORIZON, QUANTILE_LEVELS

PATTERN = 10 + 5 * np.sin(np.arange(96) * 2 * np.pi / 96)
BUILT: dict = {}


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


# The forecaster and parameters of one zero-shot model (benchmark/models ZERO_SHOT), built once per test session.
# Skips the test if the model's weights aren't in the local Hugging Face cache or its package is missing.
# So no test downloads anything.
def _zero_shot(name: str) -> tuple:
    from huggingface_hub import try_to_load_from_cache

    from benchmark.models import ZERO_SHOT
    model_id, revision, filename = ZERO_SHOT[name].weights
    if not isinstance(try_to_load_from_cache(model_id, filename, revision=revision), str):
        pytest.skip(f"{model_id} at {revision} is not in the local Hugging Face cache")
    if name not in BUILT:
        try:
            BUILT[name] = ZERO_SHOT[name].build()
        except ImportError as exc:
            pytest.skip(f"{name}: {exc}")
    return BUILT[name]


# make_contexts(n, days): n synthetic contexts (see _contexts).
@pytest.fixture
def make_contexts():
    return _contexts


# autogluon_deciles(model, hyperparameters, contexts, path): AutoGluon's deciles (see _autogluon_deciles).
@pytest.fixture
def autogluon_deciles():
    return _autogluon_deciles


# zero_shot(name): (forecaster, parameters) of a zero-shot model, built once (see _zero_shot).
@pytest.fixture
def zero_shot():
    return _zero_shot


# {UTC date: 56 nonzero buckets} for every date from first to last, except the closed ones.
def _days(first: str, last: str, closed: tuple[str, ...] = ()) -> dict[pd.Timestamp, int]:
    return {day: 56 for day in pd.date_range(first, last) if day not in pd.DatetimeIndex(closed)}


# Nonzero 15-min buckets per UTC date of two synthetic halls, shaped like the October snapshot (counts read from it):
# - both: a test day on 2025-07-08 (Eishalle a full day of 56 buckets, Schwimmhalle 1 bucket), a 9-bucket opening day
#   on 2025-10-10, full days from 2025-10-11;
# - "swim" (Schwimmhalle): open to 2026-05-16 (closed 2026-01-01 and 05-14), the trickle days of late May, the stray
#   days 2026-09-08 and 09-17 (9 buckets each), full days again from 2026-09-19 (or from `reopen`) to 2027-01-09;
#   plus one full day alone on 2026-08-10 (not in the snapshot; like Eishalle's 2025-07-08 test day, a lone open day);
# - "ice" (Eishalle): open to 2026-03-28 (closed 2026-01-01 and 02-27..03-01), then one count on 2026-09-29, and full
#   days from `reopen` when given.
def _hall_days(hall: str, reopen: Optional[str] = None) -> dict[pd.Timestamp, int]:
    counts = {pd.Timestamp("2025-07-08"): 1 if hall == "swim" else 56, pd.Timestamp("2025-10-10"): 9}
    if hall == "swim":
        counts.update(_days("2025-10-11", "2026-05-16", closed=("2026-01-01", "2026-05-14")))
        counts.update({pd.Timestamp(day): n for day, n in (
            ("2026-05-18", 9), ("2026-05-20", 9), ("2026-05-21", 11), ("2026-05-22", 14), ("2026-05-26", 26),
            ("2026-05-28", 10), ("2026-05-29", 16), ("2026-08-10", 56), ("2026-09-08", 9), ("2026-09-17", 9))})
        counts.update(_days(reopen or "2026-09-19", "2027-01-09"))
    else:
        counts.update(_days("2025-10-11", "2026-03-28",
                            closed=("2026-01-01", "2026-02-27", "2026-02-28", "2026-03-01")))
        counts[pd.Timestamp("2026-09-29")] = 1
        if reopen is not None:
            counts.update(_days(reopen, "2027-01-09"))
    return counts


# Raw hall rows: one visitors_total row per 15-min bucket in [start, until), value 5 in the first `counts[date]`
# buckets from 06:00 UTC of each date, 0 elsewhere. Rows every 15 minutes, so the fill rule changes nothing.
def _hall_raw(counts: dict[pd.Timestamp, int], until: str = "2027-01-10", start: str = "2025-07-08") -> pd.DataFrame:
    times = pd.date_range(start, until, freq="15min", inclusive="left")
    days = times.normalize()
    slot = ((times - days) // pd.Timedelta("15min")).to_numpy()
    n = np.array([counts.get(day, 0) for day in days])
    return pd.DataFrame({"created_at": times, "visitors_total": np.where((slot >= 24) & (slot < 24 + n), 5.0, 0.0)})


# hall_days(hall, reopen=None): per-date nonzero bucket counts of a synthetic hall (see _hall_days).
@pytest.fixture
def hall_days():
    return _hall_days


# hall_raw(counts, until=..., start=...): raw rows of a synthetic hall (see _hall_raw).
@pytest.fixture
def hall_raw():
    return _hall_raw
