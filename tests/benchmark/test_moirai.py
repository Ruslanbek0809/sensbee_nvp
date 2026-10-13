# Tests specific to the Moirai-2 adapter (benchmark/models/moirai_models.py); the generic zero-shot checks are in
# test_zero_shot.py. Run only in venv-moirai, where uni2ts is installed (skipped elsewhere):
#   venv-moirai/bin/python -m pytest tests/benchmark/test_moirai.py -q
#   HF_HUB_OFFLINE=1 venv-moirai/bin/python -m pytest tests/benchmark/test_moirai.py tests/benchmark/test_zero_shot.py \
#       -m integration -k moirai -q -s

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("uni2ts")

ROOT = Path(__file__).resolve().parents[2]

# In a fresh interpreter: the classes the adapter imports don't pull in uni2ts/common/env.py, whose import calls
# load_dotenv() and would read sensbee_nvp's dotenv file (the SensBee keys) into the process.
CHECK = """
import sys
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
print("uni2ts.common.env" in sys.modules)
"""


def test_moirai2_import_never_loads_the_dotenv_module():
    result = subprocess.run([sys.executable, "-c", CHECK], cwd=ROOT, capture_output=True, text=True, check=True)

    assert result.stdout.strip().splitlines()[-1] == "False"


@pytest.mark.integration
def test_moirai_refuses_contexts_of_different_lengths(make_contexts):
    from huggingface_hub import try_to_load_from_cache

    from benchmark.models import ZERO_SHOT
    from benchmark.tasks import HORIZON, QUANTILE_LEVELS
    model_id, revision, filename = ZERO_SHOT["moirai2_small"].weights
    if not isinstance(try_to_load_from_cache(model_id, filename, revision=revision), str):
        pytest.skip(f"{model_id} at {revision} is not in the local Hugging Face cache")
    forecaster, _ = ZERO_SHOT["moirai2_small"].build()

    with pytest.raises(ValueError, match="SAME LENGTH"):
        forecaster([make_contexts(1, 7)[0], make_contexts(2, 2)[1]], HORIZON, QUANTILE_LEVELS, 0)
