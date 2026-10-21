# Moirai-2.0-R-small (uni2ts 2.0.0, Moirai2Forecast) as a zero-shot harness forecaster (contract: benchmark/runner.py
# header). It runs only in venv-moirai.
# uni2ts 2.0.0 declares torch<2.5, numpy~=1.26 and scipy~=1.11, and none of them install on Python 3.13 / macOS 27.
# So it is installed with --no-deps on top of the bench stack (torch 2.10, numpy 2.3; requirements-moirai.txt).
# Every Moirai-2 result is outside the package's declared pins.
# The versions of the --no-deps packages are recorded with the settings.
#
# uni2ts/common/env.py calls load_dotenv() when it is imported, which would read sensbee_nvp's dotenv file (the SensBee
# keys) into the process.
# Only uni2ts's dataset builders and eval utilities import it, not the Moirai-2 classes used here
# (checked 2026-10-13; tests/benchmark/test_moirai.py fails if that changes).
#
# The model is loaded once per build at a pinned Hugging Face revision with strict weight loading.
# (The mixin's default is strict=False. The checkpoint and the model have the same keys, 11,387,208 parameters;
# checked 2026-10-13.)
# It is put in eval mode (its dropout is 0 anyway).
# Every call sets the context length to its contexts' length (one length per call, the runner's contract), so
# predict() never pads them.
# Moirai-2 forecasts 4 tokens of 16 steps = 64 steps in one pass.
# Steps 65-96 come from the library's recursion (uni2ts/model/moirai2/forecast.py, v2.0.0), like Chronos-Bolt.
# The recursion feeds each of the 9 quantile paths back as context and takes the quantiles of the 81 results.
# The 9 trained quantiles are the deciles, read out by level.
# Series are forecast independently and the model is deterministic, so the seed is unused.
# Weights: CC-BY-NC-4.0 (non-commercial; accepted by the owner for research, 2026-10-13).

import logging
from importlib.metadata import version

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

NO_DEPS = ("uni2ts", "jaxtyping", "wadler_lindig")  # requirements-moirai.txt


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
        columns = [trained.index(level) for level in levels]  # a level the model wasn't trained on raises
        out = []
        with torch.inference_mode(), model.hparams_context(prediction_length=horizon, context_length=len(contexts[0])):
            for start in range(0, len(contexts), batch_size):
                inputs = [c.to_numpy(dtype=np.float32) for c in contexts[start:start + batch_size]]
                out.append(model.predict(inputs)[:, columns, :horizon].transpose(0, 2, 1))
        return np.concatenate(out), ["ok"] * len(contexts)

    return forecast, {"library": "uni2ts", "model_id": model_id, "revision": revision, "device": device,
                      "dtype": "float32", "batch_size": batch_size, "patch": module.patch_size,
                      "num_predict_token": module.num_predict_token, "max_seq_len": module.max_seq_len,
                      "outside_declared_pins": "uni2ts 2.0.0 with torch 2.10, numpy 2.3 (--no-deps)",
                      "no_deps_versions": {package: version(package) for package in NO_DEPS},
                      "licence": "CC-BY-NC-4.0", "torch_threads": torch.get_num_threads(), "point_only": False}
