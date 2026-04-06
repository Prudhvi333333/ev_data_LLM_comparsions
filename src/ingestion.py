from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import re

import pandas as pd

from src.ontology import DomainOntology, normalize_for_match
from src.utils.files import file_sha256, stable_hash_dict


KB_COLUMN_MAP = {
    "Company": "company",
    "Category": "category",
    "Industry Group": "industry_group",
    "Updated Location": "updated_location",
    "Address": "address",
    "Latitude": "latitude",
    "Longitude": "longitude",
    "Primary Facility Type": "primary_facility_type",
    "EV Supply Chain Role": "ev_supply_chain_role",
    "Primary OEMs": "primary_oems",
    "Supplier or Affiliation Type": "supplier_or_affiliation_type",
    "Employment": "employment",
    "Product / Service": "product_service",
    "EV / Battery Relevant": "ev_battery_relevant",
    "Classification Method": "classification_method",
}

QUESTION_COLUMN_MAP = {
    "Num": "question_id",
    "Use Case Category": "use_case_category",
    "Question": "question",
    "Human validated answers": "reference_answer",
}


@dataclass(frozen=True)
class DatasetFingerprint:
    path: Path
    sha256: str
    row_count: int


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value).strip()
    return str(value)


def _normalize_employment(value: Any) -> float:
    raw = _normalize_text(value).replace(",", "")
    if not raw:
        return 0.0
    try:
        return float(raw)
    except ValueError:
        return 0.0


def _split_location(location: str) -> tuple[str, str]:
    parts = [part.strip() for part in str(location or "").split(",") if part.strip()]
    if not parts:
        return "", ""
    city = parts[0]
    county = ""
    for part in parts[1:]:
        if "county" in part.lower():
            county = part
            break
    return city, county


def _normalize_category(value: str) -> str:
    normalized = normalize_for_match(value)
    mapping = {
        "oem footprint": "oem footprint",
        "oem foot print": "oem footprint",
        "oem footprint footprint": "oem footprint",
        "oem supply chain": "oem supply chain",
        "tier 1": "tier 1",
        "tier 1 2": "tier 1 2",
        "tier 2 3": "tier 2 3",
        "oem": "oem",
    }
    return mapping.get(normalized, normalized)


def _normalize_facility_type(value: str) -> str:
    normalized = normalize_for_match(value)
    if "r d" in normalized or "research and development" in normalized:
        return "r d"
    if "manufacturing" in normalized and "engineering" in normalized:
        return "manufacturing engineering"
    if "manufacturing plant" in normalized:
        return "manufacturing plant"
    if "manufacturing" in normalized:
        return "manufacturing"
    return normalized


def _split_primary_oems(value: str) -> list[str]:
    normalized = normalize_for_match(value)
    if not normalized:
        return []
    alias_map = [
        ("hyundai kia rivian", ["Hyundai Kia", "Rivian Automotive"]),
        ("hyundai kia", ["Hyundai Kia"]),
        ("hyundai motor group", ["Hyundai Motor Group"]),
        ("rivian automotive", ["Rivian Automotive"]),
        ("rivian", ["Rivian Automotive"]),
        ("blue bird corp", ["Blue Bird Corp."]),
        ("blue bird", ["Blue Bird"]),
        ("club car llc", ["Club Car LLC"]),
        ("kia georgia inc", ["Kia Georgia Inc."]),
        ("mercedes benz usa llc", ["Mercedes-Benz USA LLC"]),
        ("porsche cars north america inc", ["Porsche Cars North America Inc."]),
        ("sk battery america", ["SK Battery America"]),
        ("textron specialized vehicles", ["Textron Specialized Vehicles"]),
        ("yamaha motor manufacturing corp", ["Yamaha Motor Manufacturing Corp."]),
        ("archer aviation inc", ["Archer Aviation Inc."]),
        ("multiple oems", ["Multiple OEMs"]),
    ]
    tokens: list[str] = []
    for phrase, values in alias_map:
        if phrase in normalized:
            for item in values:
                if item not in tokens:
                    tokens.append(item)
    if not tokens and value:
        tokens.append(str(value).strip())
    return tokens


def load_knowledge_base(
    path: Path,
    ontology: DomainOntology | None = None,
) -> tuple[pd.DataFrame, DatasetFingerprint]:
    frame = pd.read_excel(path).fillna("")
    missing = [column for column in KB_COLUMN_MAP if column not in frame.columns]
    if missing:
        raise ValueError(f"KB file is missing required columns: {', '.join(missing)}")

    normalized = frame.rename(columns=KB_COLUMN_MAP).copy()
    for column in normalized.columns:
        normalized[column] = normalized[column].apply(_normalize_text)

    normalized["employment"] = normalized["employment"].apply(_normalize_employment)
    city_county = normalized["updated_location"].apply(_split_location)
    normalized["city"] = city_county.apply(lambda item: item[0])
    normalized["county"] = city_county.apply(lambda item: item[1])
    normalized["company_normalized"] = normalized["company"].apply(normalize_for_match)
    normalized["category_normalized"] = normalized["category"].apply(_normalize_category)
    normalized["industry_group_normalized"] = normalized["industry_group"].apply(normalize_for_match)
    normalized["updated_location_normalized"] = normalized["updated_location"].apply(normalize_for_match)
    normalized["city_normalized"] = normalized["city"].apply(normalize_for_match)
    normalized["county_normalized"] = normalized["county"].apply(normalize_for_match)
    normalized["primary_facility_type_normalized"] = normalized["primary_facility_type"].apply(_normalize_facility_type)
    normalized["ev_supply_chain_role_normalized"] = normalized["ev_supply_chain_role"].apply(normalize_for_match)
    normalized["primary_oems_normalized"] = normalized["primary_oems"].apply(normalize_for_match)
    normalized["primary_oem_tokens"] = normalized["primary_oems"].apply(_split_primary_oems)
    normalized["primary_oem_tokens_normalized"] = normalized["primary_oem_tokens"].apply(
        lambda items: [normalize_for_match(item) for item in items]
    )
    normalized["primary_oem_tokens_text"] = normalized["primary_oem_tokens_normalized"].apply(
        lambda items: " | ".join(items)
    )
    normalized["supplier_or_affiliation_type_normalized"] = normalized["supplier_or_affiliation_type"].apply(normalize_for_match)
    normalized["product_service_normalized"] = normalized["product_service"].apply(normalize_for_match)
    normalized["ev_battery_relevant_normalized"] = normalized["ev_battery_relevant"].apply(normalize_for_match)
    normalized["classification_method_normalized"] = normalized["classification_method"].apply(normalize_for_match)
    normalized["is_supplier"] = normalized.apply(
        lambda row: bool(
            row["category_normalized"].startswith("tier")
            or row["classification_method_normalized"] == "supplier"
            or "automotive supply chain participant" in row["supplier_or_affiliation_type_normalized"]
        ),
        axis=1,
    )
    normalized["is_direct_manufacturer"] = normalized["supplier_or_affiliation_type_normalized"].str.contains(
        "original equipment manufacturer",
        na=False,
    )
    normalized["is_rd_facility"] = normalized["primary_facility_type_normalized"].eq("r d")
    normalized["search_text"] = normalized.apply(
        lambda row: " | ".join(
            [
                row["company"],
                row["category"],
                row["industry_group"],
                row["updated_location"],
                row["primary_facility_type"],
                row["ev_supply_chain_role"],
                row["primary_oems"],
                row["product_service"],
                row["ev_battery_relevant"],
            ]
        ),
        axis=1,
    )
    normalized["search_text_normalized"] = normalized["search_text"].apply(normalize_for_match)
    normalized["row_index"] = list(range(1, len(normalized) + 1))
    normalized["record_id"] = normalized.apply(
        lambda row: stable_hash_dict(
            {
                "company": row["company"],
                "category": row["category"],
                "updated_location": row["updated_location"],
                "address": row["address"],
                "row_index": int(row["row_index"]),
            }
        )[:16],
        axis=1,
    )
    if ontology is not None:
        normalized = ontology.annotate_frame(normalized)
    else:
        normalized["ontology_tags"] = [[] for _ in range(len(normalized))]
        normalized["ontology_tags_text"] = ""

    fingerprint = DatasetFingerprint(
        path=path,
        sha256=file_sha256(path),
        row_count=len(normalized),
    )
    return normalized, fingerprint


def load_questions(path: Path) -> tuple[pd.DataFrame, DatasetFingerprint]:
    frame = pd.read_excel(path).fillna("")
    missing = [column for column in QUESTION_COLUMN_MAP if column not in frame.columns]
    if missing:
        raise ValueError(f"Questions file is missing required columns: {', '.join(missing)}")

    normalized = frame.rename(columns=QUESTION_COLUMN_MAP).copy()
    for column in normalized.columns:
        normalized[column] = normalized[column].apply(_normalize_text)
    normalized["question_id"] = normalized["question_id"].astype(str)

    fingerprint = DatasetFingerprint(
        path=path,
        sha256=file_sha256(path),
        row_count=len(normalized),
    )
    return normalized, fingerprint
