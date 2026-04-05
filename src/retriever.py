from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
import re

from src.indexer import KBIndexer, _matches_local_filter, _token_ngrams, _tokenize
from src.utils.logger import get_logger


logger = get_logger("retriever")

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None


@dataclass
class _CandidateDoc:
    doc_id: str
    text: str
    metadata: dict[str, Any]
    semantic_score: float = 0.0
    bm25_score: float = 0.0
    rerank_score: float = 0.0


class HybridRetriever:
    _reranker = None
    _RELATION_PATTERNS = (
        r"connected to",
        r"\bconnection\b",
        r"linked to",
        r"\bprovid(?:e|es|ed|ing)\b.+\bto\b",
        r"\bserv(?:e|es|ed|ing)\b.+\bto\b",
        r"materials to",
        r"\bto each\b",
    )
    _BATTERY_CUSTOMER_ROLES = ("battery cell", "battery pack")
    _BATTERY_MATERIAL_TERMS = (
        "battery materials",
        "battery material",
        "electrolyte",
        "electrolytes",
        "copper foil",
        "anode",
        "anodes",
        "cathode",
        "cathodes",
        "lithium-ion",
        "lithium ion",
    )
    _POWER_COMPONENT_TERMS = (
        "dc-to-dc",
        "dc to dc",
        "converter",
        "converters",
        "capacitor",
        "capacitors",
        "power electronics",
        "ev drivetrains",
        "ev drivetrain",
    )
    _LIGHTWEIGHT_MATERIAL_TERMS = (
        "aluminum",
        "composite",
        "composites",
        "lightweight",
        "polymer",
        "polymers",
    )
    _RECYCLING_TERMS = (
        "battery recycling",
        "second-life",
        "second life",
        "end-of-life",
        "circular economy",
        "recycling",
        "recycler",
    )
    _INNOVATION_TERMS = (
        "innovation-stage",
        "innovation stage",
        "research",
        "development",
        "prototyping",
        "prototype",
        "r&d",
    )

    def __init__(self, indexer: KBIndexer, config: SimpleNamespace) -> None:
        self.indexer = indexer
        self.config = config
        self.tier_values = sorted(
            {
                str(meta.get("Category", "")).strip()
                for meta in self.indexer.metadatas
                if str(meta.get("Category", "")).strip()
            },
            key=len,
            reverse=True,
        )
        self.oem_category_values = [
            category for category in self.tier_values if category.lower().startswith("oem")
        ]
        self.role_values = sorted(
            {
                str(meta.get("EV Supply Chain Role", "")).strip()
                for meta in self.indexer.metadatas
                if str(meta.get("EV Supply Chain Role", "")).strip()
            },
            key=len,
            reverse=True,
        )
        self.oem_values = sorted(
            {
                token.strip()
                for meta in self.indexer.metadatas
                for token in str(meta.get("Primary OEMs", "")).replace("/", ";").split(";")
                if token.strip()
            },
            key=len,
            reverse=True,
        )
        self.company_values = sorted(
            {
                str(meta.get("Company", "")).strip()
                for meta in self.indexer.metadatas
                if str(meta.get("Company", "")).strip()
            },
            key=len,
            reverse=True,
        )
        self.location_values = sorted(
            {
                str(location_value).strip()
                for meta in self.indexer.metadatas
                for location_value in (
                    meta.get("Updated Location"),
                    meta.get("Location"),
                    meta.get("Updated Location City"),
                    meta.get("Updated Location County"),
                    meta.get("Location City"),
                    meta.get("Location County"),
                )
                if str(location_value or "").strip()
            },
            key=len,
            reverse=True,
        )
        self.industry_values = sorted(
            {
                str(meta.get("Industry Group", "")).strip()
                for meta in self.indexer.metadatas
                if str(meta.get("Industry Group", "")).strip()
            },
            key=len,
            reverse=True,
        )
        self.role_to_oems = self._build_role_to_oems()
        self.reranker = self._load_reranker()

    def _load_reranker(self):
        if HybridRetriever._reranker is not None:
            return HybridRetriever._reranker
        if CrossEncoder is None:
            logger.warning("CrossEncoder unavailable; using lexical rerank fallback.")
            return None
        reranker_model = self.config.retrieval.reranker_model
        try:
            HybridRetriever._reranker = CrossEncoder(reranker_model, local_files_only=True)
        except TypeError:
            try:
                HybridRetriever._reranker = CrossEncoder(reranker_model)
            except Exception as exc:
                logger.warning("Failed to load reranker (%s); using lexical fallback.", exc)
                HybridRetriever._reranker = None
        except Exception as exc:
            logger.info("Local cache load failed for reranker %s: %s", reranker_model, exc)
            try:
                HybridRetriever._reranker = CrossEncoder(reranker_model)
            except Exception as nested_exc:
                logger.warning("Failed to load reranker (%s); using lexical fallback.", nested_exc)
                HybridRetriever._reranker = None
        return HybridRetriever._reranker

    @staticmethod
    def _starts_with_any(text: str, prefixes: list[str]) -> bool:
        lowered = text.strip().lower()
        return any(lowered.startswith(prefix) for prefix in prefixes)

    @staticmethod
    def _normalize_query_text(query: str) -> str:
        return " ".join(query.lower().replace("\n", " ").split())

    @staticmethod
    def _split_oem_values(raw_value: Any) -> set[str]:
        text = str(raw_value or "").strip()
        if not text:
            return set()

        normalized = re.sub(r"[/,;|]+", ";", text)
        tokens: set[str] = set()
        for segment in normalized.split(";"):
            segment = " ".join(segment.split()).strip()
            if not segment:
                continue
            tokens.add(segment)
            if segment.lower() != "multiple oems":
                for piece in segment.split():
                    piece = piece.strip()
                    if len(piece) > 1:
                        tokens.add(piece)
        return tokens

    def _build_role_to_oems(self) -> dict[str, set[str]]:
        role_to_oems: dict[str, set[str]] = {}
        for metadata in self.indexer.metadatas:
            role = str(metadata.get("EV Supply Chain Role", "")).strip()
            if not role:
                continue
            role_to_oems.setdefault(role, set()).update(
                self._split_oem_values(metadata.get("Primary OEMs", ""))
            )
        return role_to_oems

    @staticmethod
    def _parse_employment_value(raw_value: Any) -> float:
        try:
            return float(str(raw_value or "").replace(",", "").strip() or 0.0)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _parse_threshold(query_lower: str, markers: tuple[str, ...]) -> float | None:
        for marker in markers:
            pattern = re.compile(
                rf"{re.escape(marker)}\s+([0-9][0-9,]*)"
            )
            match = pattern.search(query_lower)
            if match:
                return float(match.group(1).replace(",", ""))
        return None

    def _match_known_value(self, query_lower: str, values: list[str]) -> str | None:
        for value in values:
            if value and value.lower() in query_lower:
                return value
        return None

    def _match_known_values(self, query_lower: str, values: list[str]) -> list[str]:
        matched: list[str] = []
        for value in values:
            value_lower = value.lower()
            if value and value_lower in query_lower and value not in matched:
                matched.append(value)
        return matched

    def detect_query_intent(self, query: str) -> dict[str, Any]:
        query_lower = self._normalize_query_text(query)
        is_battery_material_query = any(
            term in query_lower for term in self._BATTERY_MATERIAL_TERMS
        )
        is_power_component_query = any(
            term in query_lower for term in self._POWER_COMPONENT_TERMS
        )
        is_lightweight_material_query = any(
            term in query_lower for term in self._LIGHTWEIGHT_MATERIAL_TERMS
        )
        is_recycling_query = any(term in query_lower for term in self._RECYCLING_TERMS)
        is_relation_query = any(
            re.search(pattern, query_lower) is not None
            for pattern in self._RELATION_PATTERNS
        )
        is_explicit_vehicle_oem_query = bool(
            re.search(r"\bvehicle assembly\s+oems?\b", query_lower)
            or re.search(r"\ball\s+vehicle assembly\s+oems?\b", query_lower)
        )
        is_global_employment_query = any(
            marker in query_lower
            for marker in (
                "highest total employment",
                "combined employment",
                "total employment across all",
                "across all companies",
                "county has the highest total employment",
            )
        )
        is_single_point_failure_query = any(
            marker in query_lower
            for marker in (
                "single-point-of-failure",
                "single point of failure",
                "served by only a single company",
                "only a single company",
            )
        )
        is_no_ev_presence_query = any(
            marker in query_lower
            for marker in (
                "no ev-specific production presence",
                "no ev specific production presence",
                "conversion-ready industrial sites",
            )
        )
        is_chemical_infra_query = any(
            marker in query_lower
            for marker in (
                "chemical manufacturing infrastructure",
                "chemical manufacturing",
            )
        )
        is_power_signal_query = any(
            marker in query_lower
            for marker in (
                "high-voltage",
                "dc-to-dc",
                "inverter",
                "motor controller",
            )
        )
        is_concentration_query = any(
            marker in query_lower
            for marker in (
                "highest concentration",
                "concentration of",
                "concentrated",
            )
        )
        if is_chemical_infra_query:
            is_battery_material_query = False
        is_vehicle_oem_mapping_query = bool(
            is_explicit_vehicle_oem_query
            and is_relation_query
            and ("tier 1" in query_lower or "tier1" in query_lower)
        )

        explicit_tier_1_2 = bool(
            re.search(r"\btier\s*1\s*/\s*2\b", query_lower)
            or re.search(r"\btier\s*1-2\b", query_lower)
            or re.search(r"\btier\s*1\s+and\s+2\b", query_lower)
        )
        if explicit_tier_1_2 and "Tier 1/2" in self.tier_values:
            tier_values = ["Tier 1/2"]
        else:
            tier_values = self._match_known_values(query_lower, self.tier_values)
        if "OEM" in tier_values and not is_explicit_vehicle_oem_query:
            tier_values = [value for value in tier_values if value != "OEM"]
        if is_explicit_vehicle_oem_query:
            for category in self.oem_category_values:
                if category not in tier_values:
                    tier_values.append(category)
        is_new_tier_reference = bool(
            re.search(r"\bfor a new tier\s*1\b", query_lower)
            or re.search(r"\bnew tier\s*1\b", query_lower)
        )
        if is_new_tier_reference and not any(
            marker in query_lower
            for marker in (
                "tier 1 suppliers",
                "tier 1 supplier",
                "tier 1 companies",
                "tier 1 company",
                "among tier 1",
                "tier 1 only",
            )
        ):
            tier_values = []
        explicit_tier_1_only = bool(
            re.search(r"\btier\s*1\b", query_lower)
            and not re.search(r"\btier\s*1\s*/\s*2\b", query_lower)
            and "tier 1/2" not in query_lower
            and "tier1/2" not in query_lower
            and "tier 1 and 2" not in query_lower
            and not is_new_tier_reference
        )
        if explicit_tier_1_only and "Tier 1" in self.tier_values:
            tier_values = ["Tier 1"]
        tier_value = tier_values[0] if tier_values else None

        company_value = self._match_known_value(query_lower, self.company_values)
        industry_values = self._match_known_values(query_lower, self.industry_values)
        if "electronic and electrical equipment" in query_lower and not industry_values:
            for value in self.industry_values:
                if "electrical equipment" in value.lower() and value not in industry_values:
                    industry_values.append(value)
        industry_value = industry_values[0] if industry_values else None

        role_values = self._match_known_values(query_lower, self.role_values)
        if not role_values:
            role_aliases = {
                "battery cell": "Battery Cell",
                "battery pack": "Battery Pack",
                "thermal management": "Thermal Management",
                "power electronics": "Power Electronics",
                "charging infrastructure": "Charging Infrastructure",
                "general automotive": "General Automotive",
                "vehicle assembly": "Vehicle Assembly",
                "raw material": "Raw Material",
                "recycling": "Recycling",
            }
            for phrase, canonical in role_aliases.items():
                if phrase in query_lower:
                    for role in self.role_values:
                        if canonical.lower() in role.lower() and role not in role_values:
                            role_values.append(role)
        if is_chemical_infra_query:
            role_values = []
        role_value = role_values[0] if role_values else None

        oem_value = self._match_known_value(query_lower, self.oem_values)
        oem_values = self._match_known_values(query_lower, self.oem_values)
        oem_alias_values: list[str] = []
        for value in oem_values:
            oem_alias_values.extend(self._split_oem_values(value))
        for alias in oem_alias_values:
            if alias not in oem_values:
                oem_values.append(alias)
        location_value = self._match_known_value(query_lower, self.location_values)
        if location_value is None and (
            "georgia" in query_lower
            or "within the state" in query_lower
            or "located in" in query_lower
        ):
            location_value = "Georgia"

        relation_target_roles: list[str] = []
        if is_relation_query:
            for role in role_values:
                role_lower = role.lower()
                if any(target_role in role_lower for target_role in self._BATTERY_CUSTOMER_ROLES):
                    relation_target_roles.append(role)
            if (
                not relation_target_roles
                and is_explicit_vehicle_oem_query
                and any(str(value).lower().startswith("tier") for value in tier_values)
            ):
                relation_target_roles = list(role_values)

        source_role_values = list(role_values)
        if relation_target_roles and len(role_values) > len(relation_target_roles):
            source_role_values = [
                role for role in role_values if role not in relation_target_roles
            ] or list(role_values)
        elif relation_target_roles and is_explicit_vehicle_oem_query:
            source_role_values = []

        if is_lightweight_material_query and not relation_target_roles:
            source_role_values = [
                role
                for role in self.role_values
                if any(marker in role.lower() for marker in ("material", "general automotive"))
            ] or source_role_values

        if is_battery_material_query and not relation_target_roles and not role_values:
            source_role_values = [
                role
                for role in self.role_values
                if any(
                    marker in role.lower()
                    for marker in (
                        "material",
                        "battery cell",
                        "battery pack",
                        "general automotive",
                    )
                )
            ] or source_role_values
        elif is_power_component_query and not relation_target_roles and not role_values:
            if " role" not in query_lower and " roles" not in query_lower:
                source_role_values = [
                    role
                    for role in self.role_values
                    if any(marker in role.lower() for marker in ("power electronics", "general automotive"))
                ] or source_role_values
        elif is_lightweight_material_query and not relation_target_roles and not role_values:
            source_role_values = [
                role
                for role in self.role_values
                if any(marker in role.lower() for marker in ("material", "general automotive"))
            ] or source_role_values
        elif is_recycling_query and not relation_target_roles and not role_values:
            source_role_values = [
                role
                for role in self.role_values
                if any(
                    marker in role.lower()
                    for marker in ("material", "battery pack", "battery cell", "general automotive", "recycling")
                )
            ] or source_role_values

        related_oems: set[str] = set()
        for role in relation_target_roles:
            related_oems.update(self.role_to_oems.get(role, set()))

        has_company_filter = company_value is not None
        if (
            is_relation_query
            and company_value is not None
            and oem_value is not None
            and company_value.lower() == oem_value.lower()
        ):
            has_company_filter = False
            company_value = None

        is_negative_ev_specific_query = any(
            marker in query_lower
            for marker in ("no ev-specific", "no ev specific", "without ev-specific", "without ev specific")
        )
        if "indirectly relevant" in query_lower:
            ev_relevance_values = ["Indirect"]
        elif is_negative_ev_specific_query:
            ev_relevance_values = ["No", "Indirect", ""]
        elif "relevant to ev drivetrains" in query_lower:
            ev_relevance_values = ["Yes", "Indirect"]
        elif "ev relevant" in query_lower or "ev-relevant" in query_lower or "ev / battery relevant" in query_lower:
            ev_relevance_values = ["Yes"]
        elif any(
            marker in query_lower
            for marker in [
                "ev-specific",
                "ev specific",
                "transition readiness",
                "some ev relevance",
                "growing their ev-specific customer base",
            ]
        ):
            ev_relevance_values = ["Yes", "Indirect"]
        elif any(
            marker in query_lower
            for marker in [
                "electric vehicle or battery supply chain",
                "ev component suppliers",
                "ev component",
                "ev drivetrains",
                "relevant to ev drivetrains",
            ]
        ) or is_battery_material_query:
            ev_relevance_values = ["Yes"]
        else:
            ev_relevance_values = []

        return {
            "has_tier_filter": tier_value is not None,
            "tier_value": tier_value,
            "tier_values": tier_values,
            "has_industry_filter": industry_value is not None,
            "industry_value": industry_value,
            "industry_values": industry_values,
            "has_role_filter": bool(source_role_values),
            "role_value": source_role_values[0] if source_role_values else None,
            "role_values": source_role_values,
            "query_role_values": role_values,
            "relation_target_roles": relation_target_roles,
            "related_oems": sorted(related_oems),
            "has_oem_filter": oem_value is not None,
            "oem_value": oem_value,
            "oem_values": oem_values,
            "has_company_filter": has_company_filter,
            "company_value": company_value,
            "is_count_query": self._starts_with_any(query_lower, ["how many", "count", "total number"]),
            "is_list_query": self._starts_with_any(
                query_lower,
                [
                    "list",
                    "list every",
                    "show all",
                    "which ",
                    "find ",
                    "identify ",
                    "name all",
                    "map all",
                    "show ",
                    "what suppliers",
                    "what companies",
                ],
            ),
            "is_location_query": location_value is not None,
            "location_value": location_value,
            "is_comparison_query": any(
                marker in query_lower
                for marker in ["compare", " vs ", "difference between", " both "]
            ),
            "requires_ev_relevant": any(
                marker in query_lower
                for marker in [
                    "ev relevant",
                    "ev-relevant",
                    "battery relevant",
                    "ev / battery relevant",
                    "ev-specific",
                    "ev specific",
                    "transition readiness",
                    "indirectly relevant",
                    "electric vehicle or battery supply chain",
                    "ev component suppliers",
                    "ev component",
                    "ev drivetrains",
                ]
            ) and not is_negative_ev_specific_query or is_battery_material_query,
            "requires_supplier_classification": bool(re.search(r"\bsuppliers?\b", query_lower)),
            "ev_relevance_values": ev_relevance_values,
            "exclude_ev_specific": is_negative_ev_specific_query,
            "is_battery_material_query": is_battery_material_query,
            "is_power_component_query": is_power_component_query,
            "is_lightweight_material_query": is_lightweight_material_query,
            "is_recycling_query": is_recycling_query,
            "is_relation_query": is_relation_query,
            "is_employment_rank_query": any(
                marker in query_lower
                for marker in [
                    "highest employment",
                    "employment size",
                    "based on employment",
                    "top 10",
                    "total employment",
                    "employ over",
                    "fewer than 200 employees",
                    "less than 200 employees",
                ]
            ),
            "min_employment_threshold": self._parse_threshold(query_lower, ("over", "more than", "above")),
            "max_employment_threshold": self._parse_threshold(query_lower, ("fewer than", "less than", "below", "under")),
            "is_innovation_query": any(
                marker in query_lower for marker in self._INNOVATION_TERMS
            ),
            "is_dual_platform_query": any(
                marker in query_lower
                for marker in ["dual-platform", "traditional oems", "ev-native oems"]
            ),
            "is_global_employment_query": is_global_employment_query,
            "is_vehicle_oem_mapping_query": is_vehicle_oem_mapping_query,
            "is_single_point_failure_query": is_single_point_failure_query,
            "is_no_ev_presence_query": is_no_ev_presence_query,
            "is_chemical_infra_query": is_chemical_infra_query,
            "is_power_signal_query": is_power_signal_query,
            "is_concentration_query": is_concentration_query,
            "requires_strict_filters": bool(
                tier_values
                or industry_values
                or source_role_values
                or oem_values
                or has_company_filter
                or (location_value is not None and location_value != "Georgia")
                or re.search(r"\bover\s+[0-9]", query_lower)
                or re.search(r"\bfewer than\s+[0-9]", query_lower)
                or re.search(r"\bonly\b", query_lower)
            ),
        }

    def _build_metadata_filter(self, intent: dict[str, Any]) -> dict[str, Any]:
        if intent.get("is_vehicle_oem_mapping_query"):
            category_values = list(dict.fromkeys([*self.oem_category_values, "Tier 1", "Tier 1/2"]))
            return {"$or": [{"Category": value} for value in category_values]}

        clauses: list[dict[str, Any]] = []

        if intent.get("has_company_filter") and intent.get("company_value"):
            clauses.append({"Company": intent["company_value"]})

        tier_values = intent.get("tier_values") or []
        if intent.get("has_tier_filter") and tier_values:
            if len(tier_values) == 1:
                clauses.append({"Category": {"$eq": tier_values[0]}})
            else:
                clauses.append({"$or": [{"Category": {"$eq": value}} for value in tier_values]})

        industry_values = intent.get("industry_values") or []
        if intent.get("has_industry_filter") and industry_values:
            if len(industry_values) == 1:
                clauses.append({"Industry Group": {"$eq": industry_values[0]}})
            else:
                clauses.append(
                    {"$or": [{"Industry Group": {"$eq": value}} for value in industry_values]}
                )

        role_values = intent.get("role_values") or []
        if intent.get("has_role_filter") and role_values:
            if len(role_values) == 1:
                clauses.append({"EV Supply Chain Role": role_values[0]})
            else:
                clauses.append(
                    {
                        "$or": [
                            {"EV Supply Chain Role": value}
                            for value in role_values
                        ]
                    }
                )

        if intent.get("has_oem_filter") and intent.get("oem_value"):
            if not intent.get("is_relation_query"):
                oem_values = intent.get("oem_values") or [intent["oem_value"]]
                if len(oem_values) == 1:
                    clauses.append({"Primary OEMs": oem_values[0]})
                else:
                    clauses.append({"$or": [{"Primary OEMs": value} for value in oem_values]})
        if intent.get("requires_ev_relevant"):
            ev_relevance_values = intent.get("ev_relevance_values") or ["Yes"]
            if len(ev_relevance_values) == 1:
                clauses.append({"EV / Battery Relevant": ev_relevance_values[0]})
            else:
                clauses.append(
                    {
                        "$or": [
                            {"EV / Battery Relevant": value}
                            for value in ev_relevance_values
                        ]
                    }
                )
        if intent.get("exclude_ev_specific"):
            clauses.append(
                {
                    "$or": [
                        {"EV / Battery Relevant": {"$eq": "No"}},
                        {"EV / Battery Relevant": {"$eq": "Indirect"}},
                        {"EV / Battery Relevant": {"$eq": ""}},
                    ]
                }
            )
        if intent.get("is_location_query") and intent.get("location_value") not in (None, "Georgia"):
            location_value = intent["location_value"]
            if str(location_value).lower().endswith("county"):
                clauses.append(
                    {
                        "$or": [
                            {"Updated Location County": location_value},
                            {"Location County": location_value},
                        ]
                    }
                )
            else:
                clauses.append(
                    {
                        "$or": [
                            {"Updated Location City": location_value},
                            {"Location City": location_value},
                            {"Updated Location": location_value},
                            {"Location": location_value},
                        ]
                    }
                )
        if intent.get("requires_supplier_classification"):
            clauses.append(
                {
                    "$or": [
                        {"Classification Method": {"$eq": "Supplier"}},
                        {"Supplier or Affiliation Type": {"$eq": "Automotive supply chain participant"}},
                    ]
                }
            )

        if not clauses:
            return {}
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    @staticmethod
    def _doc_matches_filter(metadata: dict[str, Any], metadata_filter: dict[str, Any]) -> bool:
        return _matches_local_filter(metadata, metadata_filter)

    def _bm25_candidates(
        self,
        query: str,
        candidate_pool: int,
        metadata_filter: dict[str, Any],
    ) -> list[_CandidateDoc]:
        if self.indexer.bm25 is None:
            return []

        query_tokens = _tokenize(query)
        scores = self.indexer.bm25.get_scores(query_tokens)
        ordered = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)

        candidates: list[_CandidateDoc] = []
        for doc_idx, score in ordered:
            if len(candidates) >= candidate_pool:
                break
            metadata = self.indexer.metadatas[doc_idx]
            if metadata_filter and not self._doc_matches_filter(metadata, metadata_filter):
                continue
            candidates.append(
                _CandidateDoc(
                    doc_id=self.indexer.doc_ids[doc_idx],
                    text=self.indexer.texts[doc_idx],
                    metadata=metadata,
                    bm25_score=float(score),
                )
            )
        return candidates

    def _lexical_rerank_score(
        self,
        query: str,
        candidate: _CandidateDoc,
        intent: dict[str, Any],
    ) -> float:
        query_token_list = _tokenize(query)
        doc_token_list = _tokenize(candidate.text)
        query_tokens = set(query_token_list)
        doc_tokens = set(doc_token_list)
        query_ngrams = set(_token_ngrams(query_token_list))
        doc_ngrams = set(_token_ngrams(doc_token_list))

        token_overlap = len(query_tokens & doc_tokens) / max(1, len(query_tokens))
        ngram_overlap = len(query_ngrams & doc_ngrams) / max(1, len(query_ngrams))
        bm25_norm = candidate.bm25_score / (1.0 + abs(candidate.bm25_score))

        metadata_bonus = 0.0
        metadata = candidate.metadata
        if intent.get("company_value") and str(intent["company_value"]).lower() in str(metadata.get("Company", "")).lower():
            metadata_bonus += 0.25
        tier_values = [str(value).lower() for value in intent.get("tier_values") or []]
        if tier_values and any(value in str(metadata.get("Category", "")).lower() for value in tier_values):
            metadata_bonus += 0.15
        industry_values = [str(value).lower() for value in intent.get("industry_values") or []]
        if industry_values and any(value in str(metadata.get("Industry Group", "")).lower() for value in industry_values):
            metadata_bonus += 0.20
        if intent.get("role_value") and str(intent["role_value"]).lower() in str(metadata.get("EV Supply Chain Role", "")).lower():
            metadata_bonus += 0.20
        if intent.get("oem_value") and str(intent["oem_value"]).lower() in str(metadata.get("Primary OEMs", "")).lower():
            metadata_bonus += 0.15
        oem_values = {
            str(value).strip().lower() for value in (intent.get("oem_values") or []) if str(value).strip()
        }
        if oem_values:
            candidate_oems = {
                value.lower() for value in self._split_oem_values(metadata.get("Primary OEMs", ""))
            }
            if candidate_oems & oem_values:
                metadata_bonus += 0.30
        if intent.get("location_value") and intent.get("location_value") != "Georgia":
            location_blob = f"{metadata.get('Updated Location', '')} {metadata.get('Location', '')}".lower()
            if str(intent["location_value"]).lower() in location_blob:
                metadata_bonus += 0.15
        if intent.get("requires_supplier_classification"):
            classification_method = str(metadata.get("Classification Method", "")).strip().lower()
            supplier_type = str(metadata.get("Supplier or Affiliation Type", "")).strip().lower()
            if classification_method == "supplier" or "supply chain participant" in supplier_type:
                metadata_bonus += 0.20
            else:
                metadata_bonus -= 0.10

        if intent.get("requires_ev_relevant"):
            ev_flag = str(metadata.get("EV / Battery Relevant", "")).strip().lower()
            if ev_flag == "yes":
                metadata_bonus += 0.35
            elif ev_flag == "indirect":
                metadata_bonus += 0.10
            else:
                metadata_bonus -= 0.20

        related_oems = set(intent.get("related_oems") or [])
        if related_oems:
            company_name = str(metadata.get("Company", "")).strip().lower()
            related_oems_lower = {str(related_oem).strip().lower() for related_oem in related_oems}
            if company_name and company_name in related_oems_lower:
                metadata_bonus += 0.35
            candidate_oems = self._split_oem_values(metadata.get("Primary OEMs", ""))
            if "Multiple OEMs" not in candidate_oems and "Multiple OEMs" not in related_oems:
                overlap = {
                    candidate_oem.lower()
                    for candidate_oem in candidate_oems
                } & related_oems_lower
                if overlap:
                    metadata_bonus += min(0.35, 0.15 * len(overlap))
                else:
                    metadata_bonus -= 0.05

        if intent.get("is_dual_platform_query"):
            candidate_oems = {
                value.lower() for value in self._split_oem_values(metadata.get("Primary OEMs", ""))
            }
            has_traditional = bool({"hyundai", "kia"} & candidate_oems)
            has_ev_native = "rivian" in candidate_oems
            if has_traditional and has_ev_native:
                metadata_bonus += 0.45
            elif has_ev_native:
                metadata_bonus += 0.10

        if intent.get("is_innovation_query"):
            product_text = str(metadata.get("Product / Service", "")).lower()
            if any(term in product_text for term in ("r&d", "research", "prototype", "prototyp")):
                metadata_bonus += 0.45
            elif "development" in product_text:
                metadata_bonus += 0.15

        if intent.get("is_recycling_query"):
            product_text = str(metadata.get("Product / Service", "")).lower()
            if any(term in product_text for term in ("recycl", "second-life", "second life", "end-of-life")):
                metadata_bonus += 0.45

        if intent.get("is_power_component_query"):
            product_text = str(metadata.get("Product / Service", "")).lower()
            if any(term in product_text for term in ("dc-to-dc", "converter", "capacitor", "electronics", "power electronics")):
                metadata_bonus += 0.35

        if intent.get("is_lightweight_material_query"):
            product_text = str(metadata.get("Product / Service", "")).lower()
            if any(term in product_text for term in ("aluminum", "composite", "polymer", "lightweight")):
                metadata_bonus += 0.35

        if intent.get("is_employment_rank_query"):
            employment = self._parse_employment_value(metadata.get("Employment", 0))
            metadata_bonus += min(0.35, 0.03 * (employment ** 0.25))

            min_threshold = float(intent.get("min_employment_threshold") or 0.0)
            if min_threshold > 0:
                if employment > min_threshold:
                    metadata_bonus += 0.45
                else:
                    metadata_bonus -= 0.10

            max_threshold = float(intent.get("max_employment_threshold") or 0.0)
            if max_threshold > 0:
                if 0 < employment < max_threshold:
                    metadata_bonus += 0.45
                else:
                    metadata_bonus -= 0.10

        return (
            0.35 * token_overlap
            + 0.20 * ngram_overlap
            + 0.25 * candidate.semantic_score
            + 0.15 * bm25_norm
            + metadata_bonus
        )

    def _rerank(self, query: str, candidates: list[_CandidateDoc], intent: dict[str, Any]) -> list[_CandidateDoc]:
        if not candidates:
            return []

        lexical_scores = [
            self._lexical_rerank_score(query, candidate, intent)
            for candidate in candidates
        ]

        def _minmax(values: list[float]) -> list[float]:
            if not values:
                return []
            low = min(values)
            high = max(values)
            if abs(high - low) < 1e-9:
                return [0.5 for _ in values]
            return [(value - low) / (high - low) for value in values]

        lexical_norm = _minmax(lexical_scores)

        if self.reranker is not None:
            try:
                ce_scores = [
                    float(score)
                    for score in self.reranker.predict(
                        [(query, candidate.text) for candidate in candidates]
                    )
                ]
                ce_norm = _minmax(ce_scores)

                if intent.get("is_employment_rank_query") or intent.get("is_innovation_query"):
                    ce_weight = 0.40
                    lexical_weight = 0.60
                elif intent.get("requires_ev_relevant") or intent.get("has_tier_filter"):
                    ce_weight = 0.50
                    lexical_weight = 0.50
                else:
                    ce_weight = 0.65
                    lexical_weight = 0.35

                for candidate, ce_value, lex_value in zip(candidates, ce_norm, lexical_norm):
                    candidate.rerank_score = (ce_weight * ce_value) + (lexical_weight * lex_value)
            except Exception as exc:
                logger.warning("Cross-encoder rerank failed (%s); using lexical fallback.", exc)
                for candidate, lexical_value in zip(candidates, lexical_norm):
                    candidate.rerank_score = lexical_value
        else:
            for candidate, lexical_value in zip(candidates, lexical_norm):
                candidate.rerank_score = lexical_value

        candidates.sort(key=lambda item: item.rerank_score, reverse=True)
        return candidates

    def retrieve(self, query: str) -> list[dict[str, Any]]:
        intent = self.detect_query_intent(query)
        metadata_filter = self._build_metadata_filter(intent)
        candidate_pool = int(self.config.retrieval.candidate_pool)
        top_k = int(self.config.retrieval.top_k)

        if intent["is_count_query"]:
            top_k *= 2
            candidate_pool *= 2
        elif intent["is_list_query"]:
            top_k *= 3
            candidate_pool *= 3
        if intent.get("is_employment_rank_query"):
            top_k = max(top_k, int(self.config.retrieval.top_k) * 5)
            candidate_pool = max(candidate_pool, int(self.config.retrieval.candidate_pool) * 5)
        needs_wide_relation_recall = bool(
            intent.get("is_relation_query")
            and (
                intent.get("relation_target_roles")
                or intent.get("is_vehicle_oem_mapping_query")
                or intent.get("is_battery_material_query")
            )
        )
        if (
            intent.get("is_innovation_query")
            or intent.get("is_dual_platform_query")
            or intent.get("is_recycling_query")
            or needs_wide_relation_recall
        ):
            top_k = max(top_k, int(self.config.retrieval.top_k) * 5)
            candidate_pool = max(candidate_pool, len(self.indexer.doc_ids))
        if intent.get("is_innovation_query") or (
            intent.get("is_employment_rank_query") and float(intent.get("min_employment_threshold") or 0.0) > 0
        ):
            top_k = max(top_k, 60)
        if intent.get("is_global_employment_query") or intent.get("is_vehicle_oem_mapping_query"):
            top_k = max(top_k, len(self.indexer.doc_ids))
            candidate_pool = max(candidate_pool, len(self.indexer.doc_ids))
        if (
            intent.get("is_single_point_failure_query")
            or intent.get("is_no_ev_presence_query")
            or intent.get("is_chemical_infra_query")
            or intent.get("is_dual_platform_query")
            or intent.get("is_recycling_query")
            or intent.get("is_concentration_query")
        ):
            top_k = max(top_k, len(self.indexer.doc_ids))
            candidate_pool = max(candidate_pool, len(self.indexer.doc_ids))
        if (
            intent.get("is_list_query")
            and float(intent.get("min_employment_threshold") or 0.0) > 0
        ):
            top_k = max(top_k, len(self.indexer.doc_ids))
            candidate_pool = max(candidate_pool, len(self.indexer.doc_ids))
        if intent.get("is_power_signal_query"):
            top_k = max(top_k, int(self.config.retrieval.top_k) * 8)
            candidate_pool = max(candidate_pool, int(self.config.retrieval.candidate_pool) * 8)

        filter_attempts = [metadata_filter]
        if metadata_filter and not intent.get("requires_strict_filters"):
            filter_attempts.append({})

        merged: dict[str, _CandidateDoc] = {}
        for active_filter in filter_attempts:
            semantic_candidates = self.indexer.semantic_search(
                query,
                n_results=candidate_pool,
                where=active_filter or None,
            )
            bm25_candidates = self._bm25_candidates(query, candidate_pool, active_filter)

            for candidate in semantic_candidates:
                merged[candidate.id] = _CandidateDoc(
                    doc_id=candidate.id,
                    text=candidate.text,
                    metadata=candidate.metadata,
                    semantic_score=candidate.semantic_score,
                )
            for candidate in bm25_candidates:
                if candidate.doc_id in merged:
                    merged[candidate.doc_id].bm25_score = max(
                        merged[candidate.doc_id].bm25_score,
                        candidate.bm25_score,
                    )
                else:
                    merged[candidate.doc_id] = candidate

            if merged:
                break

        reranked = self._rerank(query, list(merged.values()), intent)
        if not reranked:
            logger.warning("No retrieval candidates for query '%s'; returning empty retrieval set.", query)
            return []

        return [
            {
                "id": candidate.doc_id,
                "text": candidate.text,
                "metadata": candidate.metadata,
                "semantic_score": candidate.semantic_score,
                "rerank_score": candidate.rerank_score,
            }
            for candidate in reranked[:top_k]
        ]
