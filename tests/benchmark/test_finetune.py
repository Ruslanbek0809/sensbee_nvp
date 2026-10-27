# Integration tests for the fine-tuned Chronos-2 trainer in benchmark/models/finetune_models.py (Phase 6a, 6a.7), on
# the CPU with 2 training steps and a batch of 4 (the grid uses 250-1000 steps and 256 on the GPU):
# - full and LoRA fine-tuning give finite deciles of the right shape, and the same forecasts on a second call (the
#   model forecasts in eval mode: with dropout on, two calls would differ)
# - the same seed gives the same fine-tuned forecasts, for LoRA also when the global RNG moved in between (its A
#   matrices are drawn before the Trainer seeds)
# - learning rate 0 gives exactly the zero-shot forecasts (gate Z: the wrapper changes nothing but the weights), also
#   when training ran in bf16 as on an A100 (accelerate's bf16-autocast forward must not stay on the model; bf16 is
#   forced here on the CPU, where accelerate wraps the forward the same way)
# - micro-batches with gradient accumulation (batch 4 as 2 x 2) run the configured number of optimizer steps
# - the 8 grid configs are named in the "chronos2_ft" family; the gate config is not
# They need the Chronos-2 weights in the local Hugging Face cache and never download them. Run in the bench venv with
#   HF_HUB_OFFLINE=1 venv-bench/bin/python -m pytest tests/benchmark/test_finetune.py -m integration -q

import numpy as np
import pytest

pytestmark = pytest.mark.integration
pytest.importorskip("chronos")

import torch  # noqa: E402

from huggingface_hub import try_to_load_from_cache  # noqa: E402

from benchmark.models import CHRONOS2, FINETUNED, ZERO_SHOT  # noqa: E402
from benchmark.models.finetune_models import chronos2_finetune_trainer  # noqa: E402
from benchmark.report import model_family  # noqa: E402
from benchmark.tasks import HORIZON, QUANTILE_LEVELS  # noqa: E402

MODEL_ID, REVISION = CHRONOS2
if not isinstance(try_to_load_from_cache(MODEL_ID, "model.safetensors", revision=REVISION), str):
    pytest.skip(f"{MODEL_ID} at {REVISION} is not in the local Hugging Face cache", allow_module_level=True)


# A forecaster fine-tuned for 2 steps (batch 4) on 30 days of a noisy daily pattern, and its fit info.
def _fine_tuned(make_contexts, mode: str, lr: float, seed: int = 0):
    train, params = chronos2_finetune_trainer(MODEL_ID, REVISION, mode, lr, num_steps=2, batch_size=4)
    (item,) = make_contexts(1, 30)
    forecaster, info = train("S", [item], seed)
    return forecaster, info, params


@pytest.mark.parametrize("mode", ["full", "lora"])
def test_fine_tuned_forecasts_are_finite_and_repeat(make_contexts, mode):
    forecaster, info, params = _fine_tuned(make_contexts, mode, 1e-4)
    contexts = make_contexts(3, 28)

    batch, status = forecaster(contexts, HORIZON, QUANTILE_LEVELS, 0)
    again, _ = forecaster(contexts, HORIZON, QUANTILE_LEVELS, 0)

    assert batch.shape == (3, HORIZON, len(QUANTILE_LEVELS)) and status == ["ok"] * 3 and np.isfinite(batch).all()
    np.testing.assert_array_equal(again, batch)  # eval mode: no dropout noise
    assert info["device"] == "cpu" and params["finetune_mode"] == mode and params["context_length"] == 2688
    assert (params["lora"] is not None) == (mode == "lora")


def test_the_same_seed_gives_the_same_fine_tuned_forecasts(make_contexts):
    contexts = make_contexts(2, 28)
    first = _fine_tuned(make_contexts, "full", 1e-4)[0](contexts, HORIZON, QUANTILE_LEVELS, 0)[0]
    second = _fine_tuned(make_contexts, "full", 1e-4)[0](contexts, HORIZON, QUANTILE_LEVELS, 0)[0]
    zero_shot = ZERO_SHOT["chronos2"].build()[0](contexts, HORIZON, QUANTILE_LEVELS, 0)[0]
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, zero_shot)  # 2 steps at 1e-4 did change the model


def test_lora_with_the_same_seed_repeats_after_the_global_rng_moved(make_contexts):
    contexts = make_contexts(2, 28)
    first = _fine_tuned(make_contexts, "lora", 1e-4)[0](contexts, HORIZON, QUANTILE_LEVELS, 0)[0]
    torch.manual_seed(123)  # as after another sensor's fit in the same run
    second = _fine_tuned(make_contexts, "lora", 1e-4)[0](contexts, HORIZON, QUANTILE_LEVELS, 0)[0]
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("bf16", [False, True])
def test_learning_rate_zero_gives_the_zero_shot_forecasts(make_contexts, monkeypatch, bf16):
    from chronos.chronos2.pipeline import Chronos2Pipeline
    if bf16:
        fit = Chronos2Pipeline.fit
        monkeypatch.setattr(Chronos2Pipeline, "fit",
                            lambda self, *args, **kwargs: fit(self, *args, bf16=True, **kwargs))
    contexts = make_contexts(3, 28)
    tuned = _fine_tuned(make_contexts, "full", 0.0)[0](contexts, HORIZON, QUANTILE_LEVELS, 0)[0]
    zero_shot = ZERO_SHOT["chronos2"].build()[0](contexts, HORIZON, QUANTILE_LEVELS, 0)[0]
    np.testing.assert_array_equal(tuned, zero_shot)


def test_micro_batches_accumulate_into_the_configured_optimizer_steps(make_contexts):
    train, params = chronos2_finetune_trainer(MODEL_ID, REVISION, "full", 1e-4, num_steps=2, batch_size=4,
                                              micro_batch_size=2)
    (item,) = make_contexts(1, 30)
    forecaster, info = train("S", [item], 0)
    assert params["gradient_accumulation_steps"] == 2 and info["steps_run"] == 2
    assert np.isfinite(forecaster(make_contexts(1, 28), HORIZON, QUANTILE_LEVELS, 0)[0]).all()


def test_grid_configs_form_one_family_and_the_gate_config_is_outside_it():
    grid = [name for name in FINETUNED if name != "chronos2_ft_lr0"]
    assert len(grid) == 8 and {model_family(name) for name in grid} == {"chronos2_ft"}
    assert {model_family(name + "_pooled") for name in grid} == {"chronos2_ft"}
    assert model_family("chronos2_ft_lr0") != "chronos2_ft"
    assert all(FINETUNED[name].contexts == (28,) and FINETUNED[name].trained for name in FINETUNED)
