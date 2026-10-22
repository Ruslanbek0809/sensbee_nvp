# Tests for the Docker packaging of Chronos-2: the image must ship what the benchmark evaluated.
# - The Dockerfile downloads the model and revision pinned in benchmark/models/__init__.py.
# - requirements-tsfm.txt pins the versions of constraints-bench.txt, the benchmark's environment.

import re
from pathlib import Path

from benchmark.models import CHRONOS2

ROOT = Path(__file__).resolve().parents[2]


# The name==version pins of a requirements or constraints file, with lower-case names.
def _pins(path: Path) -> dict[str, str]:
    pins = {}
    for line in path.read_text().splitlines():
        match = re.fullmatch(r"([A-Za-z0-9_.\-]+)==(\S+)", line.strip())
        if match:
            pins[match.group(1).lower()] = match.group(2)
    return pins


def test_the_dockerfile_downloads_the_pinned_chronos2():
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert f"ARG CHRONOS2_MODEL={CHRONOS2[0]}\n" in dockerfile
    assert f"ARG CHRONOS2_REVISION={CHRONOS2[1]}\n" in dockerfile


def test_the_tsfm_requirements_match_the_benchmark_versions():
    tsfm, bench = _pins(ROOT / "requirements-tsfm.txt"), _pins(ROOT / "constraints-bench.txt")
    assert {"torch", "chronos-forecasting", "transformers"} <= set(tsfm)
    assert {name: bench.get(name) for name in tsfm} == tsfm
