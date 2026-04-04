from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pandas as pd

from src.utils.logger import get_logger
from src.utils.sample_data import ensure_sample_data


logger = get_logger("kb_loader")


def _split_location_parts(location_value: Any) -> tuple[str, str]:
    text = str(location_value or "").strip()
    if not text:
        return "", ""

    parts = [part.strip() for part in text.split(",") if part.strip()]
    city = parts[0] if parts else ""
    county = next((part for part in parts if "county" in part.lower()), "")
    return city, county


def _augment_location_metadata(row: dict[str, Any]) -> dict[str, Any]:
    location_city, location_county = _split_location_parts(row.get("Location", ""))
    updated_city, updated_county = _split_location_parts(row.get("Updated Location", ""))
    row["Location City"] = location_city
    row["Location County"] = location_county
    row["Updated Location City"] = updated_city
    row["Updated Location County"] = updated_county
    return row


def build_document_text(row: dict[str, Any]) -> str:
    location_value = row.get("Updated Location", "") or row.get("Location", "")
    return (
        f"Company: {row.get('Company', '')} | Tier: {row.get('Category', '')} | "
        f"Industry: {row.get('Industry Group', '')}\n"
        f"Location: {location_value} | Address: {row.get('Address', '')}\n"
        f"Facility: {row.get('Primary Facility Type', '')} | "
        f"EV Role: {row.get('EV Supply Chain Role', '')}\n"
        f"OEMs: {row.get('Primary OEMs', '')} | "
        f"Affiliation: {row.get('Supplier or Affiliation Type', '')}\n"
        f"Employment: {row.get('Employment', '')} | "
        f"Products: {row.get('Product / Service', '')}\n"
        f"EV Relevant: {row.get('EV / Battery Relevant', '')} | "
        f"Classification: {row.get('Classification Method', '')}"
    )


def _clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.fillna("")
    for column in cleaned.columns:
        cleaned[column] = cleaned[column].apply(
            lambda value: value.strip() if isinstance(value, str) else value
        )
    return cleaned


def load_kb(config: SimpleNamespace) -> list[dict[str, Any]]:
    ensure_sample_data(config)

    kb_df = pd.read_excel(config.paths.kb)
    kb_df = _clean_frame(kb_df)

    documents: list[dict[str, Any]] = []
    for idx, (_, row) in enumerate(kb_df.iterrows()):
        row_dict = _augment_location_metadata(row.to_dict())
        documents.append(
            {
                "id": f"company_{idx}",
                "text": build_document_text(row_dict),
                "metadata": dict(row_dict),
            }
        )

    logger.info("Loaded %s KB documents from %s", len(documents), config.paths.kb)
    return documents
