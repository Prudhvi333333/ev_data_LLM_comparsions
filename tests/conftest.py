from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingestion import load_knowledge_base
from src.ontology import DomainOntology


@pytest.fixture(scope="session")
def ontology() -> DomainOntology:
    return DomainOntology.load(ROOT / "config" / "ontology_v3.yaml")


@pytest.fixture(scope="session")
def kb_frame(ontology):
    frame, _ = load_knowledge_base(
        ROOT / "data" / "kb" / "GNEM - Auto Landscape Lat Long Updated.xlsx",
        ontology=ontology,
    )
    return frame
