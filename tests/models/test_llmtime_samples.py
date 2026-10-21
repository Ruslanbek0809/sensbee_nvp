# Tests for llmtime_samples() and nvp_llms_forecast() with mocked completions (no LLM, no network):
# - every filter rule and parse failure is counted
# - the valid samples stay on the model's scale
# - the service forecast keeps its behaviour: the median of the valid samples mapped back, the last-value fallback
#   when all are filtered, and visitor clipping

from unittest.mock import patch

import numpy as np
import pandas as pd

from src.models import nvp_llms
from src.models.serialize import create_scaler

STEPS = np.arange(96)
VALID = [500 + np.round(300 * np.sin(STEPS / 5)).astype(int), 400 + np.round(200 * np.cos(STEPS / 7)).astype(int)]
LOW = -1500 + np.round(300 * np.sin(STEPS / 5)).astype(int)  # in bounds, below the context's scaled minimum


# Serializes integers like the local LLMTime settings (precision 3, comma-separated).
def _text(values) -> str:
    return ",".join(str(int(v)) for v in values)


# One completion per outcome: two valid, a parse failure and one for each filter rule (a) to (d).
COMPLETIONS = [
    _text(VALID[0]),
    "",  # nothing to parse
    _text(VALID[0] + 6000),  # (a) above the normalized bound 5.0
    _text([500] * 96),  # (b) one unique value
    _text([500] * 94 + [501, 502]),  # (c) three unique values, std < 0.001
    _text(100 + STEPS),  # (d) a straight line
    _text(VALID[1]),
]


# A 14-day 15-min series with a daily cycle between 0 and 20.
def _series() -> pd.Series:
    index = pd.date_range("2026-02-01", periods=1344, freq="15min")
    return pd.Series(10 + 10 * np.sin(np.arange(1344) * 2 * np.pi / 96), index=index)


def test_every_parse_failure_and_filter_rule_is_counted():
    with patch.object(nvp_llms, "call_llm_completion", autospec=True, return_value=COMPLETIONS) as complete:
        result = nvp_llms.llmtime_samples(_series(), 96, provider="local", num_forecasts=7)

    assert complete.call_args.kwargs["num_forecasts"] == 7 and complete.call_args.kwargs["steps"] == 96
    assert result.counts == {"completions": 7, "parse_failed": 1, "filtered_a": 1, "filtered_b": 1, "filtered_c": 1,
                             "filtered_d": 1, "valid": 2}
    np.testing.assert_allclose(result.samples[0], VALID[0] / 1000)  # on the model's (normalized) scale
    np.testing.assert_allclose(result.samples[1], VALID[1] / 1000)
    assert result.last_value == _series().iloc[-1]


def test_a_given_completion_function_replaces_the_provider_call():
    calls = []

    def complete(**kwargs):
        calls.append(kwargs)
        return [_text(VALID[0])]

    with patch.object(nvp_llms, "call_llm_completion", autospec=True) as default:
        result = nvp_llms.llmtime_samples(_series(), 96, provider="local", num_forecasts=1, complete=complete)
    default.assert_not_called()
    assert len(calls) == 1 and calls[0]["prompt"].count(",") >= 1343 and result.counts["valid"] == 1


def test_service_forecast_is_the_median_of_the_valid_samples_mapped_back():
    series = _series()
    with patch.object(nvp_llms, "call_llm_completion", autospec=True, return_value=COMPLETIONS):
        forecast = nvp_llms.nvp_llms_forecast(series, 96, provider="local", num_forecasts=7)
    scaler = create_scaler(series.to_numpy(dtype=float), alpha=0.95, beta=0.3)
    np.testing.assert_allclose(forecast, scaler.inverse_transform(np.median(np.stack(VALID) / 1000, axis=0)))


def test_service_forecast_falls_back_to_the_last_value_and_still_clips_visitors():
    series = _series()
    with patch.object(nvp_llms, "call_llm_completion", autospec=True, return_value=COMPLETIONS[1:6]):
        fallback = nvp_llms.nvp_llms_forecast(series, 96, provider="local", num_forecasts=5)
    np.testing.assert_array_equal(fallback, np.full(96, series.iloc[-1]))

    with patch.object(nvp_llms, "call_llm_completion", autospec=True, return_value=[_text(LOW)]):
        raw = nvp_llms.nvp_llms_forecast(series, 96, provider="local", num_forecasts=1)
        clipped = nvp_llms.nvp_llms_forecast(series, 96, provider="local", num_forecasts=1,
                                             column_name="visitors_total")
    assert raw.min() < 0 and clipped.min() == 0
    np.testing.assert_array_equal(clipped, np.clip(raw, 0, None))
