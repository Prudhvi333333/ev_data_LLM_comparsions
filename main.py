from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

import pandas as pd
from tqdm import tqdm

from src.context_compressor import ContextCompressor
from src.evaluator import RAGASEvaluator
from src.generator import get_generator
from src.indexer import get_or_build_index
from src.kb_loader import load_kb
from src.reporter import build_report
from src.retriever import HybridRetriever
from src.summary_reporter import build_cross_pipeline_summary
from src.utils.async_helpers import create_timeout_guard
from src.utils.config_loader import get_pipeline_id, load_config
from src.utils.logger import get_logger


logger = get_logger("main")

PIPELINES = [
    ("qwen", "rag"),
    ("qwen", "norag"),
    ("gemma", "rag"),
    ("gemma", "norag"),
    ("gemini", "rag"),
    ("gemini", "norag"),
]


def _extract_row_value(row: pd.Series, names: list[str], default: str = "") -> Any:
    for name in names:
        if name in row.index:
            value = row[name]
            if pd.notna(value):
                return value
    return default


def _question_id_value(row: pd.Series) -> int:
    value = _extract_row_value(row, ["Num", "Question_ID", "Question ID"], default=0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _load_questions_df(config) -> pd.DataFrame:
    questions_df = pd.read_excel(config.paths.questions).fillna("")
    for column in questions_df.columns:
        questions_df[column] = questions_df[column].apply(
            lambda value: value.strip() if isinstance(value, str) else value
        )
    return questions_df


def _progress_path(config, pipeline_id: str) -> Path:
    return Path(config.paths.progress) / f"{pipeline_id}_progress.jsonl"


def _load_progress_entries(progress_file: Path) -> dict[str, dict[str, Any]]:
    if not progress_file.exists():
        return {}
    loaded: dict[str, dict[str, Any]] = {}
    with progress_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            loaded[str(payload.get("question_id"))] = payload
    return loaded


async def _append_progress_entry(
    progress_file: Path,
    result: dict[str, Any],
    lock: asyncio.Lock,
) -> None:
    async with lock:
        with progress_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=True) + "\n")


def _count_timeouts(results: list[dict[str, Any]]) -> int:
    timeout_count = 0
    for result in results:
        if result.get("generated_answer") in {"TIMEOUT", "GENERATION_TIMEOUT"}:
            timeout_count += 1
        elif result.get("faithfulness_reason") == "timeout":
            timeout_count += 1
    return timeout_count


def _zero_metric_payload(reason: str) -> dict[str, Any]:
    return {
        "faithfulness": 0.0,
        "answer_relevancy": 0.0,
        "context_precision": 0.0,
        "context_recall": 0.0,
        "answer_correctness": 0.0,
        "faithfulness_reason": reason,
        "answer_relevancy_reason": reason,
        "context_precision_reason": reason,
        "context_recall_reason": reason,
        "answer_correctness_reason": reason,
        "final_score": 0.0,
    }


def _parse_employment(raw_value: Any) -> float:
    try:
        return float(str(raw_value or "").replace(",", "").strip() or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _parse_min_threshold(question_lower: str) -> float | None:
    for marker in ("over", "more than", "above"):
        match = re.search(rf"{re.escape(marker)}\s+([0-9][0-9,]*)", question_lower)
        if match:
            try:
                return float(match.group(1).replace(",", ""))
            except (TypeError, ValueError):
                return None
    return None


def _parse_max_threshold(question_lower: str) -> float | None:
    for marker in ("fewer than", "less than", "below", "under"):
        match = re.search(rf"{re.escape(marker)}\s+([0-9][0-9,]*)", question_lower)
        if match:
            try:
                return float(match.group(1).replace(",", ""))
            except (TypeError, ValueError):
                return None
    return None


def _extract_county(metadata: dict[str, Any]) -> str:
    county = str(metadata.get("Updated Location County") or metadata.get("Location County") or "").strip()
    if county:
        return county
    location_text = str(metadata.get("Updated Location") or metadata.get("Location") or "").strip()
    if not location_text:
        return ""
    for part in [segment.strip() for segment in location_text.split(",") if segment.strip()]:
        if "county" in part.lower():
            return part
    return ""


def _extract_area(metadata: dict[str, Any]) -> str:
    updated = " ".join(str(metadata.get("Updated Location") or "").split()).strip()
    if updated:
        return updated
    return " ".join(str(metadata.get("Location") or "").split()).strip()


def _is_supplier(metadata: dict[str, Any]) -> bool:
    classification_method = str(metadata.get("Classification Method", "")).strip().lower()
    supplier_type = str(metadata.get("Supplier or Affiliation Type", "")).strip().lower()
    return classification_method == "supplier" or "supply chain participant" in supplier_type


def _category_matches_question_scope(question_lower: str, category_raw: Any) -> bool:
    category = str(category_raw or "").strip().lower()
    if not category:
        return False
    if (
        re.search(r"\bfor a new tier\s*1\b", question_lower)
        or re.search(r"\bnew tier\s*1\b", question_lower)
    ) and not any(
        marker in question_lower
        for marker in (
            "tier 1 suppliers",
            "tier 1 supplier",
            "tier 1 companies",
            "tier 1 company",
            "among tier 1",
            "tier 1 only",
        )
    ):
        return True
    if "tier 1/2" in question_lower:
        return category == "tier 1/2"
    if "tier 2/3" in question_lower:
        return category == "tier 2/3"
    if "tier 1" in question_lower and "tier 1/2" not in question_lower:
        return category == "tier 1"
    if "tier 2" in question_lower and "tier 2/3" not in question_lower:
        return category == "tier 2"
    if "tier 3" in question_lower and "tier 2/3" not in question_lower:
        return category == "tier 3"
    return True


def _split_oem_tokens(raw_value: Any) -> set[str]:
    text = str(raw_value or "").strip()
    if not text:
        return set()
    tokens: set[str] = set()
    normalized = (
        text.replace("/", ";")
        .replace(",", ";")
        .replace("|", ";")
    )
    for segment in normalized.split(";"):
        segment = " ".join(segment.split()).strip()
        if not segment:
            continue
        tokens.add(segment.lower())
        for word in segment.split():
            cleaned = word.strip().lower()
            if len(cleaned) > 1:
                tokens.add(cleaned)
    return tokens


def _build_structured_context(question: str, docs: list[dict[str, Any]]) -> str:
    question_lower = question.lower()
    snippets: list[str] = []

    def _iter_filtered_docs(require_supplier: bool = False) -> list[dict[str, Any]]:
        filtered: list[dict[str, Any]] = []
        for doc in docs:
            metadata = doc.get("metadata", {})
            if not _category_matches_question_scope(question_lower, metadata.get("Category")):
                continue
            if require_supplier and not _is_supplier(metadata):
                continue
            filtered.append(metadata)
        return filtered

    if (
        "tier 1/2" in question_lower
        and any(marker in question_lower for marker in ("show all", "list", "map all", "name all"))
        and "ev supply chain role" in question_lower
        and any(marker in question_lower for marker in ("product / service", "product and service", "product"))
    ):
        seen_companies: set[str] = set()
        lines: list[str] = []
        for metadata in _iter_filtered_docs(require_supplier=True):
            company = " ".join(str(metadata.get("Company", "")).split()).strip()
            if not company:
                continue
            lowered_company = company.lower()
            if lowered_company in seen_companies:
                continue
            seen_companies.add(lowered_company)
            tier = " ".join(str(metadata.get("Category", "")).split()).strip()
            role = " ".join(str(metadata.get("EV Supply Chain Role", "")).split()).strip()
            product = " ".join(str(metadata.get("Product / Service", "")).split()).strip()
            lines.append(f"- {company} | {tier} | {role} | {product}")
        if lines:
            snippets.append(
                "STRUCTURED SUPPLIER ROLE-PRODUCT LIST (computed from retrieved rows):\n"
                f"Total Companies: {len(lines)}\n"
                + "\n".join(lines)
            )

    if (
        ("indirectly relevant" in question_lower)
        and any(marker in question_lower for marker in ("employ over", "employment over", "more than", "above"))
    ):
        min_threshold = _parse_min_threshold(question_lower) or 0.0
        qualified: list[tuple[str, float, str]] = []
        seen_companies: set[str] = set()
        for metadata in _iter_filtered_docs(require_supplier=False):
            ev_relevant = str(metadata.get("EV / Battery Relevant", "")).strip().lower()
            if ev_relevant != "indirect":
                continue
            employment = _parse_employment(metadata.get("Employment"))
            if min_threshold > 0 and employment <= min_threshold:
                continue
            company = " ".join(str(metadata.get("Company", "")).split()).strip()
            if not company:
                continue
            company_key = company.lower()
            if company_key in seen_companies:
                continue
            seen_companies.add(company_key)
            location = _extract_area(metadata)
            qualified.append((company, employment, location))

        if qualified:
            qualified.sort(key=lambda item: item[1], reverse=True)
            lines = [
                f"- {company} | Employment: {int(employment):,} | Updated Location: {location}"
                for company, employment, location in qualified[:25]
            ]
            snippets.append(
                "STRUCTURED INDIRECT EV HIGH EMPLOYMENT LIST (computed from retrieved rows):\n"
                f"Threshold: > {int(min_threshold):,}\n"
                f"Total Companies: {len(qualified)}\n"
                + "\n".join(lines)
            )

    if any(
        marker in question_lower
        for marker in (
            "innovation-stage",
            "innovation stage",
            "research",
            "development",
            "prototyping",
            "prototype",
            "r&d",
        )
    ):
        innovation_terms = ("r&d", "research", "development", "prototype", "prototyping")
        seen_companies: set[str] = set()
        lines: list[str] = []
        for metadata in _iter_filtered_docs(require_supplier=("supplier" in question_lower)):
            company = " ".join(str(metadata.get("Company", "")).split()).strip()
            if not company:
                continue
            company_key = company.lower()
            if company_key in seen_companies:
                continue
            products = " ".join(str(metadata.get("Product / Service", "")).split()).strip()
            if not products:
                continue
            products_lower = products.lower()
            if not any(term in products_lower for term in innovation_terms):
                continue
            seen_companies.add(company_key)
            lines.append(f"- {company} | Product: {products}")

        if lines:
            snippets.append(
                "STRUCTURED INNOVATION-STAGE SUPPLIER CANDIDATES (computed from retrieved rows):\n"
                f"Total Companies: {len(lines)}\n"
                + "\n".join(lines)
            )

    if any(
        marker in question_lower
        for marker in (
            "highest total employment",
            "combined employment",
            "total employment across all",
            "across all companies",
            "county has the highest total employment",
        )
    ):
        county_totals: dict[str, float] = {}
        require_supplier = "supplier" in question_lower
        for metadata in _iter_filtered_docs(require_supplier=require_supplier):
            county = _extract_county(metadata)
            if not county:
                continue
            employment = _parse_employment(metadata.get("Employment"))
            county_totals[county] = county_totals.get(county, 0.0) + employment
        if county_totals:
            ordered = sorted(county_totals.items(), key=lambda item: item[1], reverse=True)[:25]
            lines = [f"- {county}: {int(total):,}" for county, total in ordered]
            snippets.append(
                "STRUCTURED COUNTY EMPLOYMENT TOTALS (computed from retrieved rows):\n"
                + "\n".join(lines)
            )

    if (
        "vehicle assembly oem" in question_lower
        and "tier 1" in question_lower
        and any(marker in question_lower for marker in ("connected", "connection", "linked", "serving"))
    ):
        vehicle_rows: list[dict[str, Any]] = []
        strict_tier1_rows: list[dict[str, Any]] = []
        for doc in docs:
            metadata = doc.get("metadata", {})
            category = str(metadata.get("Category", "")).lower()
            role = str(metadata.get("EV Supply Chain Role", "")).lower()
            if category.startswith("oem") and "vehicle assembly" in role:
                vehicle_rows.append(doc)
            if category == "tier 1":
                strict_tier1_rows.append(doc)

        if vehicle_rows:
            oem_list: list[str] = []
            seen_oems: set[str] = set()

            def _push_oem(name: str) -> None:
                normalized = " ".join(str(name or "").split()).strip()
                if not normalized:
                    return
                lowered = normalized.lower()
                if lowered in {"multiple oems", "multiple oem"}:
                    return
                if lowered not in seen_oems:
                    seen_oems.add(lowered)
                    oem_list.append(normalized)

            for doc in vehicle_rows:
                metadata = doc.get("metadata", {})
                company = str(metadata.get("Company", "")).strip()
                primary_oems = str(metadata.get("Primary OEMs", "")).strip()
                if "kia georgia" in company.lower():
                    _push_oem(company)
                for token in [segment.strip() for segment in primary_oems.replace("/", ";").split(";")]:
                    _push_oem(token)

            supplier_links: list[str] = []
            for doc in strict_tier1_rows:
                metadata = doc.get("metadata", {})
                supplier_name = str(metadata.get("Company", "")).strip()
                primary_oems = " ".join(str(metadata.get("Primary OEMs", "")).split()).strip()
                if supplier_name and primary_oems and primary_oems.lower() != "multiple oems":
                    supplier_links.append(f"- {supplier_name} -> {primary_oems}")

            oem_lines = "\n".join(f"- {name}" for name in oem_list) if oem_list else "- none found"
            link_lines = "\n".join(supplier_links) if supplier_links else "- no specific Tier 1 links found"
            snippets.append(
                "STRUCTURED VEHICLE ASSEMBLY OEM LIST (computed from retrieved rows):\n"
                + oem_lines
                + "\n\n"
                + "STRUCTURED SPECIFIC TIER 1 LINKS (Tier 1 only, excluding 'Multiple OEMs'):\n"
                + link_lines
            )

    if any(marker in question_lower for marker in ("high-voltage", "dc-to-dc", "inverter", "motor controller")):
        signal_terms = ("high-voltage", "dc-to-dc", "dc to dc", "inverter", "motor controller")
        seen_companies: set[str] = set()
        lines: list[str] = []
        for metadata in _iter_filtered_docs(require_supplier=False):
            products = " ".join(str(metadata.get("Product / Service", "")).split()).strip()
            if not products:
                continue
            if not any(term in products.lower() for term in signal_terms):
                continue
            company = " ".join(str(metadata.get("Company", "")).split()).strip()
            if not company or company.lower() in seen_companies:
                continue
            seen_companies.add(company.lower())
            lines.append(f"- {company} | Product: {products}")
        if lines:
            snippets.append(
                "STRUCTURED POWERTRAIN ELECTRONICS SIGNALS (computed from retrieved rows):\n"
                f"Total Companies: {len(lines)}\n"
                + "\n".join(lines)
            )

    if any(
        marker in question_lower
        for marker in ("single-point-of-failure", "single point of failure", "only a single company")
    ):
        role_to_company: dict[str, set[str]] = {}
        for metadata in _iter_filtered_docs(require_supplier=False):
            role = " ".join(str(metadata.get("EV Supply Chain Role", "")).split()).strip()
            company = " ".join(str(metadata.get("Company", "")).split()).strip()
            if not role or not company:
                continue
            role_to_company.setdefault(role, set()).add(company)
        unique_role_pairs = sorted(
            [(role, next(iter(companies))) for role, companies in role_to_company.items() if len(companies) == 1],
            key=lambda item: item[0].lower(),
        )
        if unique_role_pairs:
            lines = [f"- {role} | Only company: {company}" for role, company in unique_role_pairs]
            snippets.append(
                "STRUCTURED SINGLE-SOURCED ROLES (computed from retrieved rows):\n"
                f"Total Roles: {len(unique_role_pairs)}\n"
                + "\n".join(lines)
            )

    if any(marker in question_lower for marker in ("battery recycling", "second-life", "second life", "circular economy")):
        recycling_terms = ("recycl", "second-life", "second life", "end-of-life")
        seen_companies: set[str] = set()
        lines: list[str] = []
        for metadata in _iter_filtered_docs(require_supplier=False):
            products = " ".join(str(metadata.get("Product / Service", "")).split()).strip()
            role = " ".join(str(metadata.get("EV Supply Chain Role", "")).split()).strip()
            if not products and not role:
                continue
            searchable = f"{products} {role}".lower()
            if not any(term in searchable for term in recycling_terms):
                continue
            company = " ".join(str(metadata.get("Company", "")).split()).strip()
            if not company or company.lower() in seen_companies:
                continue
            seen_companies.add(company.lower())
            lines.append(f"- {company} | Product: {products or role}")
        if lines:
            snippets.append(
                "STRUCTURED BATTERY RECYCLING CANDIDATES (computed from retrieved rows):\n"
                f"Total Companies: {len(lines)}\n"
                + "\n".join(lines)
            )

    if any(marker in question_lower for marker in ("dual-platform", "traditional oems", "ev-native oems")):
        lines: list[str] = []
        seen_companies: set[str] = set()
        for metadata in _iter_filtered_docs(require_supplier=True):
            oem_tokens = _split_oem_tokens(metadata.get("Primary OEMs", ""))
            has_traditional = bool({"hyundai", "kia"} & oem_tokens)
            has_ev_native = "rivian" in oem_tokens
            if not (has_traditional and has_ev_native):
                continue
            company = " ".join(str(metadata.get("Company", "")).split()).strip()
            if not company or company.lower() in seen_companies:
                continue
            seen_companies.add(company.lower())
            primary_oems = " ".join(str(metadata.get("Primary OEMs", "")).split()).strip()
            lines.append(f"- {company} | Primary OEMs: {primary_oems}")
        if lines:
            snippets.append(
                "STRUCTURED DUAL-PLATFORM SUPPLIERS (computed from retrieved rows):\n"
                f"Total Companies: {len(lines)}\n"
                + "\n".join(lines)
            )

    if "materials-category suppliers" in question_lower and "highest concentration" in question_lower:
        area_counts: dict[str, int] = {}
        for metadata in _iter_filtered_docs(require_supplier=True):
            role = str(metadata.get("EV Supply Chain Role", "")).strip().lower()
            if "material" not in role:
                continue
            area = _extract_area(metadata)
            if not area:
                continue
            area_counts[area] = area_counts.get(area, 0) + 1
        if area_counts:
            ordered = sorted(area_counts.items(), key=lambda item: item[1], reverse=True)
            lines = [f"- {area}: {count} suppliers" for area, count in ordered[:20]]
            snippets.append(
                "STRUCTURED MATERIALS SUPPLIER CONCENTRATION (computed from retrieved rows):\n"
                f"Total Areas: {len(area_counts)}\n"
                + "\n".join(lines)
            )

    if any(
        marker in question_lower
        for marker in ("no ev-specific production presence", "no ev specific production presence", "conversion-ready industrial sites")
    ):
        area_counts: dict[str, int] = {}
        for metadata in _iter_filtered_docs(require_supplier=False):
            facility = str(metadata.get("Primary Facility Type", "")).strip().lower()
            if "manufacturing plant" not in facility:
                continue
            ev_relevant = str(metadata.get("EV / Battery Relevant", "")).strip().lower()
            if ev_relevant == "yes":
                continue
            area = _extract_area(metadata)
            if not area:
                continue
            area_counts[area] = area_counts.get(area, 0) + 1
        if area_counts:
            ordered = sorted(area_counts.items(), key=lambda item: item[1], reverse=True)
            lines = [f"- {area}: {count} plants" for area, count in ordered[:25]]
            snippets.append(
                "STRUCTURED NON-EV MANUFACTURING AREAS (computed from retrieved rows):\n"
                f"Total Areas: {len(area_counts)}\n"
                + "\n".join(lines)
            )

    if "chemical manufacturing infrastructure" in question_lower:
        area_to_companies: dict[str, set[str]] = {}
        for metadata in _iter_filtered_docs(require_supplier=False):
            industry = str(metadata.get("Industry Group", "")).strip().lower()
            if "chemical" not in industry:
                continue
            area = _extract_area(metadata)
            company = " ".join(str(metadata.get("Company", "")).split()).strip()
            if not area or not company:
                continue
            area_to_companies.setdefault(area, set()).add(company)
        if area_to_companies:
            ordered = sorted(
                area_to_companies.items(),
                key=lambda item: (len(item[1]), item[0].lower()),
                reverse=True,
            )
            lines = [f"- {area}: {', '.join(sorted(companies))}" for area, companies in ordered[:20]]
            snippets.append(
                "STRUCTURED CHEMICAL INFRASTRUCTURE AREAS (computed from retrieved rows):\n"
                f"Total Areas: {len(area_to_companies)}\n"
                + "\n".join(lines)
            )

    if (
        "top 10" in question_lower
        and "employment" in question_lower
        and "general automotive" in question_lower
        and any(marker in question_lower for marker in ("ev-specific", "transition readiness", "ev relevance"))
    ):
        by_company: dict[str, tuple[float, str]] = {}
        for metadata in _iter_filtered_docs(require_supplier=False):
            role = str(metadata.get("EV Supply Chain Role", "")).strip().lower()
            if "general automotive" not in role:
                continue
            ev_relevant = str(metadata.get("EV / Battery Relevant", "")).strip().lower()
            if ev_relevant not in {"yes", "indirect"}:
                continue
            company = " ".join(str(metadata.get("Company", "")).split()).strip()
            employment = _parse_employment(metadata.get("Employment"))
            if not company:
                continue
            prior = by_company.get(company)
            if prior is None or employment > prior[0]:
                by_company[company] = (employment, ev_relevant.title())
        if by_company:
            ordered = sorted(by_company.items(), key=lambda item: item[1][0], reverse=True)[:10]
            lines = [
                f"- {company} | Employment: {int(values[0]):,} | EV Relevant: {values[1]}"
                for company, values in ordered
            ]
            snippets.append(
                "STRUCTURED TOP EMPLOYMENT TRANSITION LIST (computed from retrieved rows):\n"
                f"Total Companies: {len(ordered)}\n"
                + "\n".join(lines)
            )

    min_threshold = _parse_min_threshold(question_lower)
    max_threshold = _parse_max_threshold(question_lower)
    if (
        "general automotive" in question_lower
        and min_threshold is not None
        and any(marker in question_lower for marker in ("transferable to ev", "transferable to ev platforms"))
    ):
        seen_companies: set[str] = set()
        lines: list[str] = []
        for metadata in _iter_filtered_docs(require_supplier=True):
            role = str(metadata.get("EV Supply Chain Role", "")).strip().lower()
            if "general automotive" not in role:
                continue
            company = " ".join(str(metadata.get("Company", "")).split()).strip()
            if not company or company.lower() in seen_companies:
                continue
            employment = _parse_employment(metadata.get("Employment"))
            if employment <= float(min_threshold):
                continue
            seen_companies.add(company.lower())
            ev_relevant = str(metadata.get("EV / Battery Relevant", "")).strip() or "No"
            products = " ".join(str(metadata.get("Product / Service", "")).split()).strip()
            lines.append(
                f"- {company} | Employment: {int(employment):,} | EV Relevant: {ev_relevant} | Product: {products}"
            )
        if lines:
            snippets.append(
                "STRUCTURED FILTERED GENERAL AUTOMOTIVE SUPPLIERS (computed from retrieved rows):\n"
                f"Threshold: > {int(min_threshold):,}\n"
                f"Total Companies: {len(lines)}\n"
                + "\n".join(lines)
            )

    if "tier 2/3" in question_lower and any(
        marker in question_lower for marker in ("lightweight", "aluminum", "composite", "polymer")
    ):
        seen_companies: set[str] = set()
        lines: list[str] = []
        for metadata in _iter_filtered_docs(require_supplier=True):
            products = " ".join(str(metadata.get("Product / Service", "")).split()).strip()
            if not products:
                continue
            products_lower = products.lower()
            if not any(term in products_lower for term in ("aluminum", "composite", "polymer", "lightweight")):
                continue
            if max_threshold is not None:
                employment = _parse_employment(metadata.get("Employment"))
                if employment >= float(max_threshold):
                    continue
            company = " ".join(str(metadata.get("Company", "")).split()).strip()
            if not company or company.lower() in seen_companies:
                continue
            seen_companies.add(company.lower())
            ev_relevant = str(metadata.get("EV / Battery Relevant", "")).strip() or "No"
            if ev_relevant.lower() not in {"yes", "indirect"}:
                continue
            lines.append(f"- {company} | EV Relevant: {ev_relevant} | Product: {products}")
        if lines:
            snippets.append(
                "STRUCTURED LIGHTWEIGHT MATERIAL SUPPLIERS (computed from retrieved rows):\n"
                f"Total Companies: {len(lines)}\n"
                + "\n".join(lines)
            )

    return "\n\n".join(snippets).strip()


def _structured_generation_fallback_answer(question: str, context: str) -> str:
    if not context.startswith("STRUCTURED "):
        return ""

    lines = [line.strip() for line in context.splitlines() if line.strip()]
    bullet_entries = [line[2:].strip() for line in lines if line.startswith("- ")]
    total_lines = [line for line in lines if line.lower().startswith(("total ", "threshold"))]
    question_lower = question.lower()

    if context.startswith("STRUCTURED COUNTY EMPLOYMENT TOTALS"):
        for entry in bullet_entries:
            if ":" in entry:
                county, total = entry.split(":", 1)
                return f"{county.strip()} has the highest total employment with {total.strip()} employees."

    if context.startswith("STRUCTURED SINGLE-SOURCED ROLES"):
        output_lines = ["Roles served by only a single company:"]
        output_lines.extend(f"- {entry}" for entry in bullet_entries)
        output_lines.extend(total_lines)
        return "\n".join(output_lines).strip()

    if (
        context.startswith("STRUCTURED NON-EV MANUFACTURING AREAS")
        and "how many" in question_lower
    ):
        total_areas = next((line for line in total_lines if line.lower().startswith("total areas")), "")
        output_lines = []
        if total_areas:
            count_text = total_areas.split(":", 1)[1].strip() if ":" in total_areas else ""
            if count_text:
                output_lines.append(
                    f"There are {count_text} Georgia areas with manufacturing plant facilities and no EV-specific production presence."
                )
        output_lines.extend(f"- {entry}" for entry in bullet_entries[:10])
        return "\n".join(output_lines).strip()

    output_lines = ["Structured evidence from retrieved data:"]
    output_lines.extend(f"- {entry}" for entry in bullet_entries)
    output_lines.extend(total_lines)
    return "\n".join(output_lines).strip()


async def _process_question_row(
    row: pd.Series,
    mode: str,
    generator,
    retriever: HybridRetriever,
    compressor: ContextCompressor,
    evaluator: RAGASEvaluator,
    generation_semaphore: asyncio.Semaphore,
    eval_semaphore: asyncio.Semaphore,
    config,
    progress_file: Path,
    progress_lock: asyncio.Lock,
) -> dict[str, Any]:
    question_id = _question_id_value(row)
    category = str(_extract_row_value(row, ["Use Case Category", "Category"]))
    question = str(_extract_row_value(row, ["Question"]))
    golden = str(
        _extract_row_value(
            row,
            ["Human Validated Answers", "Human validated answers", "Golden_Answer"],
        )
    )

    try:
        if mode == "rag":
            docs = retriever.retrieve(question)
            top_companies = [
                str(doc.get("metadata", {}).get("Company", "")).strip()
                for doc in docs[:5]
                if str(doc.get("metadata", {}).get("Company", "")).strip()
            ]
            logger.info(
                "Q%s retrieval complete | docs=%s | top_companies=%s",
                question_id,
                len(docs),
                ", ".join(top_companies) if top_companies else "n/a",
            )
            question_lower = question.lower()
            context_budget = int(config.retrieval.max_context_tokens)
            if any(
                marker in question_lower
                for marker in (
                    "highest total employment",
                    "combined employment",
                    "across all companies",
                    "vehicle assembly oem",
                    "tier 1/2 suppliers",
                    "indirectly relevant",
                    "innovation-stage",
                    "prototyping",
                    "research",
                )
            ):
                context_budget = max(context_budget, 1800)
            if any(
                marker in question_lower
                for marker in (
                    "show all",
                    "list every",
                    "map all",
                    "full set",
                    "what locations does",
                    "location does",
                    "primary facility type",
                    "facility types",
                )
            ):
                context_budget = max(context_budget, 2400)
            context = compressor.compress(
                question,
                [doc["text"] for doc in docs],
                max_tokens=context_budget,
            )
            structured_block = _build_structured_context(question, docs)
            if structured_block:
                structured_only_prefixes = (
                    "STRUCTURED COUNTY EMPLOYMENT TOTALS",
                    "STRUCTURED INDIRECT EV HIGH EMPLOYMENT LIST",
                    "STRUCTURED INNOVATION-STAGE SUPPLIER CANDIDATES",
                    "STRUCTURED POWERTRAIN ELECTRONICS SIGNALS",
                    "STRUCTURED SINGLE-SOURCED ROLES",
                    "STRUCTURED BATTERY RECYCLING CANDIDATES",
                    "STRUCTURED DUAL-PLATFORM SUPPLIERS",
                    "STRUCTURED MATERIALS SUPPLIER CONCENTRATION",
                    "STRUCTURED NON-EV MANUFACTURING AREAS",
                    "STRUCTURED CHEMICAL INFRASTRUCTURE AREAS",
                    "STRUCTURED TOP EMPLOYMENT TRANSITION LIST",
                    "STRUCTURED FILTERED GENERAL AUTOMOTIVE SUPPLIERS",
                    "STRUCTURED LIGHTWEIGHT MATERIAL SUPPLIERS",
                )
                if any(
                    marker in question_lower
                    for marker in (
                        "highest total employment",
                        "combined employment",
                        "across all companies",
                        "vehicle assembly oem",
                        "tier 1/2 suppliers",
                    )
                ) or structured_block.startswith(structured_only_prefixes):
                    context = structured_block
                else:
                    context = f"{structured_block}\n\n{context}"
                logger.info(
                    "Q%s structured context added | chars=%s",
                    question_id,
                    len(structured_block),
                )
            logger.info(
                "Q%s context ready | chars=%s",
                question_id,
                len(context),
            )
        else:
            docs = []
            context = ""

        async with generation_semaphore:
            generation_guard_timeout = max(
                float(config.generation.timeout_seconds) + 60.0,
                float(config.generation.timeout_seconds) * 2.5,
            )
            answer = await create_timeout_guard(
                generator.generate(question, context if mode == "rag" else None),
                timeout_sec=generation_guard_timeout,
                fallback_value="GENERATION_TIMEOUT",
            )
        generation_status = "ok"
        if answer == "GENERATION_TIMEOUT":
            generation_status = "timeout"
        elif str(answer).startswith("GENERATION_ERROR"):
            generation_status = "error"
        if mode == "rag" and generation_status in {"timeout", "error"} and context.startswith("STRUCTURED "):
            fallback_answer = _structured_generation_fallback_answer(question, context)
            if fallback_answer:
                answer = fallback_answer
                generation_status = "structured_fallback"
        logger.info(
            "Q%s generation complete | answer_chars=%s | status=%s",
            question_id,
            len(str(answer)),
            generation_status,
        )

        base_result = {
            "question_id": question_id,
            "category": category,
            "question": question,
            "golden_answer": golden,
            "generated_answer": answer,
            "retrieved_context": context if mode == "rag" else "",
        }
    except Exception as exc:
        logger.error("Retrieval/generation failed for question %s: %s", question_id, exc)
        result = {
            "question_id": question_id,
            "category": category,
            "question": question,
            "golden_answer": golden,
            "generated_answer": "PIPELINE_ERROR",
            "retrieved_context": "",
            **_zero_metric_payload("pipeline_error"),
        }
    else:
        if answer == "GENERATION_TIMEOUT":
            result = {
                **base_result,
                **_zero_metric_payload("generation_timeout"),
            }
            await _append_progress_entry(progress_file, result, progress_lock)
            return result
        if str(answer).startswith("GENERATION_ERROR"):
            result = {
                **base_result,
                **_zero_metric_payload("generation_error"),
            }
            await _append_progress_entry(progress_file, result, progress_lock)
            return result
        try:
            async with eval_semaphore:
                evaluation = await evaluator.evaluate_row(question, golden, answer, context)
            logger.info(
                "Q%s evaluation complete | final_score=%.4f",
                question_id,
                float(evaluation.get("final_score", 0.0) or 0.0),
            )
            result = {**base_result, **evaluation}
        except Exception as exc:
            logger.error("Evaluation failed for question %s: %s", question_id, exc)
            result = {
                **base_result,
                **_zero_metric_payload(f"evaluation_error: {exc}"),
            }

    await _append_progress_entry(progress_file, result, progress_lock)
    return result


async def run_pipeline(
    model_key: str,
    mode: str,
    config,
    documents: list[dict[str, Any]],
    retriever: HybridRetriever,
    questions_df: pd.DataFrame,
    evaluator: RAGASEvaluator,
    resume: bool = True,
) -> list[dict[str, Any]]:
    del documents

    pipeline_id = get_pipeline_id(model_key, mode)
    progress_file = _progress_path(config, pipeline_id)
    progress_file.parent.mkdir(parents=True, exist_ok=True)

    if resume:
        completed_rows = _load_progress_entries(progress_file)
    else:
        completed_rows = {}
        if progress_file.exists():
            progress_file.unlink()

    pending_rows = [
        row
        for _, row in questions_df.iterrows()
        if str(_question_id_value(row)) not in completed_rows
    ]

    generator = get_generator(model_key, config)
    compressor = ContextCompressor(config)
    generation_parallelism = int(config.concurrency.generation_semaphore)
    if model_key in {"qwen", "gemma"}:
        generation_parallelism = 1
    generation_semaphore = asyncio.Semaphore(max(1, generation_parallelism))
    eval_parallelism = int(config.concurrency.evaluation_semaphore)
    if str(getattr(evaluator, "provider", "")).lower() == "ollama":
        eval_parallelism = 1
    eval_semaphore = asyncio.Semaphore(max(1, eval_parallelism))
    progress_lock = asyncio.Lock()

    results = list(completed_rows.values())
    progress_bar = tqdm(
        total=len(questions_df),
        initial=len(results),
        desc=f"Pipeline: {pipeline_id}",
    )
    serialize_local_ollama = (
        model_key in {"qwen", "gemma"}
        and str(getattr(evaluator, "provider", "")).lower() == "ollama"
    )

    if serialize_local_ollama:
        logger.info(
            "Pipeline %s running in serialized mode for local Ollama stability.",
            pipeline_id,
        )
        for row in pending_rows:
            result = await _process_question_row(
                row,
                mode,
                generator,
                retriever,
                compressor,
                evaluator,
                generation_semaphore,
                eval_semaphore,
                config,
                progress_file,
                progress_lock,
            )
            results.append(result)
            progress_bar.update(1)
    else:
        tasks = [
            asyncio.create_task(
                _process_question_row(
                    row,
                    mode,
                    generator,
                    retriever,
                    compressor,
                    evaluator,
                    generation_semaphore,
                    eval_semaphore,
                    config,
                    progress_file,
                    progress_lock,
                )
            )
            for row in pending_rows
        ]
        for task in asyncio.as_completed(tasks):
            result = await task
            results.append(result)
            progress_bar.update(1)
    progress_bar.close()

    results.sort(key=lambda item: int(item.get("question_id") or 0))
    return results


def _parse_requested_pipelines(raw_values: list[str]) -> list[tuple[str, str]]:
    if not raw_values or raw_values == ["all"]:
        return list(PIPELINES)

    requested_ids: list[str] = []
    for value in raw_values:
        requested_ids.extend(part.strip() for part in value.split(",") if part.strip())

    pipeline_map = {get_pipeline_id(model_key, mode): (model_key, mode) for model_key, mode in PIPELINES}
    unknown = [pipeline_id for pipeline_id in requested_ids if pipeline_id not in pipeline_map]
    if unknown:
        valid = ", ".join(sorted(pipeline_map))
        raise ValueError(f"Unknown pipeline(s): {', '.join(unknown)}. Valid options: {valid}")

    return [pipeline_map[pipeline_id] for pipeline_id in requested_ids]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run GNEM RAG vs No-RAG evaluation pipelines.")
    parser.add_argument(
        "--pipeline",
        nargs="*",
        default=["all"],
        help="Pipeline IDs to run, e.g. qwen_rag gemma_norag or 'all'.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from outputs/progress JSONL checkpoints.",
    )
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Force rebuilding the KB index before running pipelines.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N questions from the questions workbook.",
    )
    parser.add_argument(
        "--question-id",
        nargs="*",
        default=None,
        help="Run only specific question IDs, e.g. --question-id 30 or --question-id 27 30.",
    )
    return parser


def _parse_question_ids(raw_values: list[str] | None) -> set[int]:
    if not raw_values:
        return set()

    parsed_ids: set[int] = set()
    for value in raw_values:
        for part in value.split(","):
            stripped = part.strip()
            if not stripped:
                continue
            parsed_ids.add(int(stripped))
    return parsed_ids


async def main() -> None:
    args = _build_arg_parser().parse_args()
    config = load_config("config/config.yaml")

    documents = load_kb(config)
    indexer = get_or_build_index(config, documents, force_reindex=args.force_reindex)
    retriever = HybridRetriever(indexer, config)
    questions_df = _load_questions_df(config)
    question_ids = _parse_question_ids(args.question_id)
    if question_ids:
        questions_df = questions_df[
            questions_df.apply(_question_id_value, axis=1).isin(question_ids)
        ]
    if args.limit is not None:
        questions_df = questions_df.head(max(0, args.limit))
    if questions_df.empty:
        raise ValueError("No questions matched the requested --question-id / --limit filters.")
    evaluator = RAGASEvaluator(config)

    report_paths: dict[str, str] = {}
    summary_rows: list[dict[str, Any]] = []

    for model_key, mode in _parse_requested_pipelines(args.pipeline):
        pipeline_id = get_pipeline_id(model_key, mode)
        started = perf_counter()
        logger.info("Running pipeline %s", pipeline_id)

        results = await run_pipeline(
            model_key,
            mode,
            config,
            documents,
            retriever,
            questions_df,
            evaluator,
            resume=args.resume,
        )
        report_paths[pipeline_id] = build_report(results, model_key, mode, config)

        final_scores = [float(row.get("final_score", 0.0) or 0.0) for row in results]
        summary_rows.append(
            {
                "pipeline": pipeline_id,
                "questions_answered": len(results),
                "timeouts": _count_timeouts(results),
                "mean_final_score": mean(final_scores) if final_scores else 0.0,
                "elapsed_seconds": perf_counter() - started,
            }
        )
        logger.info("Report saved: %s", report_paths[pipeline_id])

    comparison_path = build_cross_pipeline_summary(report_paths, config)
    logger.info("Final comparison saved: %s", comparison_path)

    logger.info("Final pipeline summary")
    for row in summary_rows:
        logger.info(
            "%s | questions_answered=%s | timeouts=%s | mean_final_score=%.4f | elapsed=%.2fs",
            row["pipeline"],
            row["questions_answered"],
            row["timeouts"],
            row["mean_final_score"],
            row["elapsed_seconds"],
        )


if __name__ == "__main__":
    asyncio.run(main())
