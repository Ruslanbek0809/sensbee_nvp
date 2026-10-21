# Tests specific to the Moirai-2 adapter (benchmark/models/moirai_models.py); the generic zero-shot checks are in
# test_zero_shot.py. Run only in venv-moirai, where uni2ts is installed (skipped elsewhere):
#   venv-moirai/bin/python -m pytest tests/benchmark/test_moirai.py -q
#   HF_HUB_OFFLINE=1 venv-moirai/bin/python -m pytest tests/benchmark/test_zero_shot.py -m integration -k moirai -q -s

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("uni2ts")

ROOT = Path(__file__).resolve().parents[2]

# Runs in a fresh interpreter. The classes the adapter imports must not pull in uni2ts/common/env.py.
# Importing that file calls load_dotenv(). That would read sensbee_nvp's dotenv file (SensBee keys) into the process.
CHECK = """
import sys
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
print("uni2ts.common.env" in sys.modules)
"""


def test_moirai2_import_never_loads_the_dotenv_module():
    result = subprocess.run([sys.executable, "-c", CHECK], cwd=ROOT, capture_output=True, text=True, check=True)

    assert result.stdout.strip().splitlines()[-1] == "False"

