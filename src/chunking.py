from __future__ import annotations

from typing import Any

import pandas as pd

from src.schemas import KnowledgeChunk
from src.utils.files import stable_hash_dict


CHUNK_FIELD_ORDER = [
    ("Company", "company"),
    ("Category", "category"),
    ("Industry Group", "industry_group"),
    ("Updated Location", "updated_location"),
    ("City", "city"),
    ("County", "county"),
    ("Address", "address"),
    ("Primary Facility Type", "primary_facility_type"),
    ("EV Supply Chain Role", "ev_supply_chain_role"),
    ("Primary OEMs", "primary_oems"),
    ("Supplier or Affiliation Type", "supplier_or_affiliation_type"),
    ("Employment", "employment"),
    ("Product / Service", "product_service"),
    ("EV / Battery Relevant", "ev_battery_relevant"),
    ("Classification Method", "classification_method"),
    ("Primary OEM Tokens", "primary_oem_tokens"),
    ("Ontology Tags", "ontology_tags_text"),
]


def render_row_chunk(row: pd.Series) -> str:
    lines = []
    for label, key in CHUNK_FIELD_ORDER:
        value = row.get(key, "")
        if key == "employment":
            value = int(float(value or 0))
        if isinstance(value, list):
            value = ", ".join(str(item) for item in value)
        lines.append(f"{label}: {value}")
    return "\n".join(lines)


def build_chunks(kb_frame: pd.DataFrame, corpus_hash: str) -> list[KnowledgeChunk]:
    chunks: list[KnowledgeChunk] = []
    for _, row in kb_frame.iterrows():
        metadata: dict[str, Any] = row.to_dict()
        text = render_row_chunk(row)
        chunk_id = stable_hash_dict(
            {
                "corpus_hash": corpus_hash,
                "record_id": metadata["record_id"],
                "text": text,
            }
        )[:16]
        chunks.append(
            KnowledgeChunk(
                chunk_id=chunk_id,
                record_id=str(metadata["record_id"]),
                text=text,
                metadata=metadata,
            )
        )
    return chunks
