"""Pytest path + shared pipeline fixture."""

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import NeurovaNLPPipeline


@pytest.fixture(scope="session")
def nlp() -> NeurovaNLPPipeline:
    try:
        return NeurovaNLPPipeline.load()
    except FileNotFoundError:
        pytest.skip("Model weights are not trained yet.")
