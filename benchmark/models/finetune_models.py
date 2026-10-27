# Chronos-2 fine-tuned on SensBee history (Phase 6a, step 6a.7, RQ4) as a harness trainer (contract:
# benchmark/runner.py header). The trainer copies the pinned zero-shot model, fine-tunes the copy on the training
# items before the cutoff with the library's own Chronos2Pipeline.fit, and forecasts with it exactly like the
# zero-shot adapter (chronos_models.chronos2_pipeline_forecaster: cross_learning=False, the deciles read out).
#
# chronos-forecasting 2.3.2 (chronos2/pipeline.py, dataset.py; checked 2026-10-27):
# - fit() builds a fresh Chronos2Model, which is in training mode, and nothing on the predict path calls .eval().
#   Chronos-2 has dropout 0.1, so the trainer calls .eval() itself; otherwise every forecast would be noisy.
# - With bf16 (on an A100), accelerate replaces the model's forward with a bf16-autocast wrapper (accelerator.py
#   prepare_model, accelerate 1.15.0) and nothing undoes it, so the fine-tuned model would forecast in bf16. The
#   trainer removes the wrapper, so forecasts are float32 like the zero-shot ones.
# - get_peft_model draws LoRA's random A matrices before the Trainer seeds, so the trainer seeds first itself.
# - Library defaults kept and recorded: linear learning-rate decay, no warmup, AdamW (fused), min_past = the horizon,
#   LoRA r 8 / alpha 16 on the attention q/k/v/o and the output layer. On a GPU with compute capability >= 8 (A100),
#   fit() trains in bf16 with tf32 and restores tf32 when it returns; forecasts are float32.
# - Training windows: a series is drawn uniformly per item (not per step), a window end uniformly within it, with
#   the global numpy RNG. The seed goes to the Hugging Face Trainer (TrainingArguments.seed), which seeds numpy and
#   torch when it starts. Each series is its own group, so pooled items never attend to each other.
# - No validation set (no early stopping): the number of steps is a grid value. The checkpoint folder is deleted
#   after the fit; the forecasts are the result, and a test run refits with its own cutoff.
# Values only (no covariates). Context 28 days, the training context length too.

import logging
import tempfile
import time

import numpy as np
import pandas as pd

from benchmark.models.chronos_models import chronos2_pipeline_forecaster
from benchmark.tasks import BUCKETS_PER_DAY, HORIZON

logger = logging.getLogger(__name__)

CONTEXT_STEPS = 28 * BUCKETS_PER_DAY  # 2,688: the 28-day context, for training windows and forecasts
BATCH_SIZE = 256                      # the library default
LORA_DEFAULT = {"r": 8, "lora_alpha": 16,
                "target_modules": ["self_attention.q", "self_attention.v", "self_attention.k", "self_attention.o",
                                   "output_patch_embedding.output_layer"]}


# The 6a.7 grid: mode x learning rate x steps (8 configs), named so report.model_family groups them as "chronos2_ft".
def finetune_grid() -> dict[str, dict]:
    grid = {}
    for mode, rates in (("full", (1e-6, 1e-5)), ("lora", (1e-5, 1e-4))):
        for lr in rates:
            for steps in (250, 1000):
                grid[f"chronos2_ft_tuned_{mode}-lr{lr:g}-s{steps}"] = {
                    "mode": mode, "learning_rate": lr, "num_steps": steps}
    return grid


# Builds the trainer for one fine-tuning config and returns it with the parameters recorded in run.json.
def chronos2_finetune_trainer(model_id: str, revision: str, mode: str, learning_rate: float, num_steps: int,
                              batch_size: int = BATCH_SIZE):
    params = {"model_id": model_id, "revision": revision, "finetune_mode": mode, "learning_rate": learning_rate,
              "num_steps": num_steps, "batch_size": batch_size, "context_length": CONTEXT_STEPS,
              "prediction_length": HORIZON, "lr_scheduler": "linear", "warmup_steps": 0, "optim": "adamw_torch_fused",
              "min_past": HORIZON, "lora": LORA_DEFAULT if mode == "lora" else None, "validation": None,
              "cross_learning": False, "point_only": False}

    def train(sensor: str, items: list[pd.Series], seed: int):
        import torch
        from chronos.chronos2.pipeline import Chronos2Pipeline
        from transformers import TrainerCallback, set_seed
        if not items:
            raise ValueError(f"NO TRAINING ITEMS FOR {sensor}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        base = Chronos2Pipeline.from_pretrained(model_id, revision=revision, device_map=device, dtype=torch.float32)
        losses = []

        # Keeps the training loss the Trainer logs every logging_steps.
        class LossLog(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs and "loss" in logs:
                    losses.append([state.global_step, float(logs["loss"])])

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        set_seed(seed)  # LoRA's A matrices are drawn before the Trainer seeds
        with tempfile.TemporaryDirectory(prefix="chronos2_ft_") as workdir:
            tuned = base.fit([item.to_numpy(dtype=np.float32) for item in items], prediction_length=HORIZON,
                             finetune_mode=mode, learning_rate=learning_rate, num_steps=num_steps,
                             batch_size=batch_size, context_length=CONTEXT_STEPS, output_dir=workdir,
                             callbacks=[LossLog()], remove_printer_callback=True, seed=seed, logging_steps=50,
                             disable_tqdm=True)
        tuned.model.__dict__.pop("forward", None)  # accelerate's bf16-autocast forward: back to the model's own
        tuned.model.__dict__.pop("_original_forward", None)
        tuned.model.eval()
        logger.info(f"FINE-TUNED Chronos-2 ({mode}, lr {learning_rate:g}, {num_steps} steps) for {sensor} "
                    f"in {time.perf_counter() - started:.0f} s")
        info = {"device": device, "train_s": round(time.perf_counter() - started, 3), "losses": losses,
                "bf16": device == "cuda" and torch.cuda.get_device_capability()[0] >= 8,
                "peak_gpu_mib": round(torch.cuda.max_memory_allocated() / 2**20) if device == "cuda" else None,
                "gpu": torch.cuda.get_device_name() if device == "cuda" else None}
        return chronos2_pipeline_forecaster(tuned, batch_size), info

    return train, params
