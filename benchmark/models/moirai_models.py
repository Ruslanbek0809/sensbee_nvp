# Moirai-2.0-R-small (uni2ts 2.0.0, Moirai2Forecast) as a zero-shot harness forecaster (contract: benchmark/runner.py
# header). Runs only in venv-moirai: uni2ts 2.0.0 declares torch<2.5, numpy~=1.26 and scipy~=1.11, none of which
# install on Python 3.13 / macOS 27, so it is installed with --no-deps on top of the bench stack (torch 2.10, numpy
# 2.3; requirements-moirai.txt). Every Moirai-2 result is outside the package's declared pins.
#
# uni2ts/common/env.py calls load_dotenv() when it is imported, which would read sensbee_nvp's dotenv file (the SensBee
# keys) into the process; only uni2ts's dataset builders and eval utilities import it, not the Moirai-2 classes used
# here (checked 2026-10-13; tests/benchmark/test_moirai.py fails if that changes).
# The model is loaded once per build at a pinned Hugging Face revision with strict weight loading (the mixin's default
# is strict=False; the checkpoint and the model have the same keys, 11,387,208 parameters; checked 2026-10-13) and put in eval mode (its dropout is 0 anyway). Every call sets the
# context length to its contexts' length, so they are never padded; contexts of different lengths in one call raise
# (the runner then retries the origins one by one). Moirai-2 forecasts 4 tokens of 16 steps = 64 steps in one pass;
# steps 65-96 come from the library's recursion, which feeds each of the 9 quantile paths back as context and takes
# the quantiles of the 81 results (uni2ts/model/moirai2/forecast.py, v2.0.0), like Chronos-Bolt. The 9 trained
# quantiles are the deciles, read out by level. Series are forecast independently and the model is deterministic, so
# the seed is unused. Weights: CC-BY-NC-4.0 (non-commercial; accepted by the owner for research, 2026-10-13).

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Builds a Moirai-2 forecaster for one model id and revision, and returns it with the effective settings.
def moirai2_forecaster(model_id: str, revision: str, device: str = "cpu", batch_size: int = 64):
    import torch
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
    module = Moirai2Module.from_pretrained(model_id, revision=revision, strict=True)
    model = Moirai2Forecast(prediction_length=1, target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0,
                            context_length=1, module=module).to(device)
    model.eval()
    trained = list(module.quantile_levels)

    def forecast(contexts: list[pd.Series], horizon: int, levels: tuple[float, ...],
                 seed: int) -> tuple[np.ndarray, list[str]]:
        lengths = {len(c) for c in contexts}
        if len(lengths) > 1:
            raise ValueError("MOIRAI-2 NEEDS CONTEXTS OF THE SAME LENGTH IN ONE CALL (SHORT ONES WOULD BE PADDED)")
        columns = [trained.index(level) for level in levels]  # a level the model wasn't trained on raises
        out = []
        with torch.inference_mode(), model.hparams_context(prediction_length=horizon, context_length=lengths.pop()):
            for start in range(0, len(contexts), batch_size):
                inputs = [c.to_numpy(dtype=np.float32) for c in contexts[start:start + batch_size]]
                out.append(model.predict(inputs)[:, columns, :horizon].transpose(0, 2, 1))
        return np.concatenate(out).astype(float), ["ok"] * len(contexts)

    return forecast, {"library": "uni2ts", "model_id": model_id, "revision": revision, "device": device,
                      "dtype": "float32", "batch_size": batch_size, "patch": module.patch_size,
                      "num_predict_token": module.num_predict_token, "max_seq_len": module.max_seq_len,
                      "outside_declared_pins": "uni2ts 2.0.0 with torch 2.10, numpy 2.3 (--no-deps)",
                      "licence": "CC-BY-NC-4.0", "torch_threads": torch.get_num_threads(),
                      "point_only": False}
