from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import re

import pandas as pd

from src.ontology import DomainOntology, normalize_for_match


@dataclass
class StructuredQueryPlan:
    route_name: str
    question_type: str
    filters: list[dict[str, Any]]
    supplier_only: bool
    requires_structured_context: bool
    group_by: list[str]
    aggregate: str | None
    aggregate_field: str | None
    sort_by: str | None
    sort_desc: bool
    top_n: int | None
    requested_fields: list[str]
    dedupe_by: str | None = None
    ontology_buckets: list[str] = field(default_factory=list)
    ontology_match_mode: str = "any"
    preferred_match_fields: list[str] = field(default_factory=list)
    target_company: str | None = None
    target_oem: str | None = None
    answer_schema: list[str] = field(default_factory=list)
    exhaustive: bool = False
    use_retrieval_support: bool = True
    support_record_ids: list[str] = field(default_factory=list)
    retrieval_query: str | None = None
    route_notes: list[str] = field(default_factory=list)


class QueryRouter:
    def __init__(
        self,
        kb_frame: pd.DataFrame,
        ontology: DomainOntology,
        top_n_default: int = 5,
        ontology_match_mode: str = "any",
    ) -> None:
        self.kb_frame = kb_frame.copy()
        self.ontology = ontology
        self.top_n_default = top_n_default
        self.ontology_match_mode = ontology_match_mode
        self.company_lookup = self._build_lookup(self.kb_frame["company"].tolist())
        self.county_lookup = self._build_lookup(self.kb_frame["county"].tolist())
        self.city_lookup = self._build_lookup(self.kb_frame["city"].tolist())
        self.facility_lookup = self._build_lookup(self.kb_frame["primary_facility_type"].tolist())
        self.role_lookup = self._build_lookup(self.kb_frame["ev_supply_chain_role"].tolist())
        self.primary_oem_lookup = self._build_lookup(
            [token for tokens in self.kb_frame["primary_oem_tokens"].tolist() for token in tokens]
        )

    @staticmethod
    def _build_lookup(values: list[Any]) -> list[tuple[str, str]]:
        unique: dict[str, str] = {}
        for value in values:
            raw = str(value or "").strip()
            normalized = normalize_for_match(raw)
            if raw and normalized and normalized not in unique:
                unique[normalized] = raw
        return sorted(unique.items(), key=lambda item: len(item[0]), reverse=True)

    def _find_known_value(self, question: str, lookup: list[tuple[str, str]]) -> str | None:
        normalized_question = normalize_for_match(question)
        for normalized_value, raw_value in lookup:
            if normalized_value and normalized_value in normalized_question:
                return raw_value
        return None

    def _match_category_filters(self, question: str) -> list[str]:
        lowered = normalize_for_match(question)
        if re.search(r"\btier 1 2\b|\btier 1 and 2\b|\btier 1\/2\b", lowered):
            return ["Tier 1/2"]
        if re.search(r"\btier 2 3\b|\btier 2\/3\b", lowered):
            return ["Tier 2/3"]
        if re.search(r"\btier 1\b", lowered):
            return ["Tier 1"]
        if "oem supply chain" in lowered:
            return ["OEM Supply Chain"]
        if "oem footprint" in lowered:
            return ["OEM Footprint", "OEM (Footprint)", "OEM Footprint"]
        return []

    def _build_filters(self, question: str) -> tuple[list[dict[str, Any]], bool, list[str]]:
        normalized_question = normalize_for_match(question)
        filters: list[dict[str, Any]] = []
        route_notes: list[str] = []
        supplier_only = "supplier" in normalized_question

        categories = self._match_category_filters(question)
        if categories:
            filters.append(
                {
                    "field": "category_normalized",
                    "op": "in" if len(categories) > 1 else "eq",
                    "value": [normalize_for_match(item) for item in categories]
                    if len(categories) > 1
                    else normalize_for_match(categories[0]),
                }
            )
            route_notes.append(f"category={categories}")

        company = self._find_known_value(question, self.company_lookup)
        if company:
            filters.append({"field": "company_normalized", "op": "eq", "value": normalize_for_match(company)})
            route_notes.append(f"company={company}")

        county = self._find_known_value(question, self.county_lookup)
        if county:
            filters.append({"field": "county_normalized", "op": "eq", "value": normalize_for_match(county)})
            route_notes.append(f"county={county}")

        city = self._find_known_value(question, self.city_lookup)
        if city and not county:
            filters.append({"field": "city_normalized", "op": "eq", "value": normalize_for_match(city)})
            route_notes.append(f"city={city}")

        facility = self._find_known_value(question, self.facility_lookup)
        if facility:
            filters.append({"field": "primary_facility_type_normalized", "op": "contains", "value": normalize_for_match(facility)})
            route_notes.append(f"facility={facility}")

        if "direct manufacturer" in normalized_question:
            filters.append({"field": "supplier_or_affiliation_type_normalized", "op": "contains", "value": "original equipment manufacturer"})
            supplier_only = False
            route_notes.append("direct_manufacturer")

        if "indirectly relevant" in normalized_question or "indirect relevant" in normalized_question:
            filters.append({"field": "ev_battery_relevant_normalized", "op": "eq", "value": "indirect"})
            route_notes.append("ev_relevance=indirect")

        if "multiple oems" in normalized_question:
            filters.append({"field": "primary_oems_normalized", "op": "contains", "value": "multiple oems"})
            route_notes.append("primary_oems=multiple_oems")

        if "employment over" in normalized_question or "employ over" in normalized_question:
            match = re.search(r"(?:employment over|employ over)\s+([0-9,]+)", normalized_question)
            if match:
                filters.append({"field": "employment", "op": "gte", "value": float(match.group(1).replace(",", ""))})
                route_notes.append(f"employment>={match.group(1)}")

        if "fewer than" in normalized_question:
            match = re.search(r"fewer than\s+([0-9,]+)", normalized_question)
            if match:
                filters.append({"field": "employment", "op": "lt", "value": float(match.group(1).replace(",", ""))})
                route_notes.append(f"employment<{match.group(1)}")

        if "over 1 000" in normalized_question or "over 1000" in normalized_question:
            filters.append({"field": "employment", "op": "gt", "value": 1000.0})
            route_notes.append("employment>1000")

        return filters, supplier_only, route_notes

    def _target_oem(self, question: str) -> str | None:
        normalized_question = normalize_for_match(question)
        if "rivian" in normalized_question:
            return "Rivian Automotive"
        return self._find_known_value(question, self.primary_oem_lookup)

    @staticmethod
    def _mentions_rd(question: str) -> bool:
        lowered = question.lower()
        return bool(
            re.search(r"\br\s*(?:&|and)\s*d\b", lowered)
            or "research and development" in lowered
            or "research development" in lowered
        )

    @staticmethod
    def _extract_top_n(question: str, default: int = 5) -> int:
        normalized_question = normalize_for_match(question)
        digit_match = re.search(r"\btop\s+([0-9]+)\b", normalized_question)
        if digit_match:
            return int(digit_match.group(1))
        if "which three" in normalized_question:
            return 3
        if "top ten" in normalized_question:
            return 10
        return default

    def route(self, question: str) -> StructuredQueryPlan:
        filters, supplier_only, route_notes = self._build_filters(question)
        normalized_question = normalize_for_match(question)
        ontology_buckets = self.ontology.question_buckets(question)
        target_oem = self._target_oem(question)
        target_company = self._find_known_value(question, self.company_lookup)

        if "single point of failure" in normalized_question or "only a single company" in normalized_question:
            return StructuredQueryPlan(
                route_name="sole_source_roles",
                question_type="sole_source_roles",
                filters=filters,
                supplier_only=False,
                requires_structured_context=True,
                group_by=["ev_supply_chain_role"],
                aggregate="nunique",
                aggregate_field="company",
                sort_by="ev_supply_chain_role",
                sort_desc=False,
                top_n=None,
                requested_fields=["ev_supply_chain_role", "company"],
                answer_schema=["ev_supply_chain_role", "company"],
                exhaustive=True,
                use_retrieval_support=False,
                route_notes=route_notes + ["sole_source_roles"],
            )

        if (
            target_oem
            and ("supplier network" in normalized_question or "linked to" in normalized_question or "linked" in normalized_question)
        ):
            filters = [
                filter_spec
                for filter_spec in filters
                if not (
                    filter_spec.get("field") == "company_normalized"
                    and filter_spec.get("value") == normalize_for_match(target_oem)
                )
            ]
            filters.append({"field": "primary_oem_tokens_text", "op": "contains", "value": normalize_for_match(target_oem)})
            return StructuredQueryPlan(
                route_name="oem_supplier_network",
                question_type="oem_supplier_network",
                filters=filters,
                supplier_only=False,
                requires_structured_context=True,
                group_by=[],
                aggregate=None,
                aggregate_field=None,
                sort_by="category_normalized",
                sort_desc=False,
                top_n=None,
                requested_fields=["company", "category", "ev_supply_chain_role", "primary_oems", "product_service"],
                dedupe_by="company",
                target_oem=target_oem,
                answer_schema=["company", "category", "ev_supply_chain_role", "primary_oems", "product_service"],
                exhaustive=True,
                use_retrieval_support=True,
                retrieval_query=f"{target_oem} supplier network Georgia",
                route_notes=route_notes + [f"target_oem={target_oem}"],
            )

        if (
            "highest total employment" in normalized_question
            or "combined employment" in normalized_question
            or "total employment across all companies" in normalized_question
        ):
            return StructuredQueryPlan(
                route_name="county_employment_ranking",
                question_type="county_employment_ranking",
                filters=filters,
                supplier_only=supplier_only and "across all companies" not in normalized_question,
                requires_structured_context=True,
                group_by=["county"],
                aggregate="sum",
                aggregate_field="employment",
                sort_by="employment",
                sort_desc=True,
                top_n=self.top_n_default,
                requested_fields=["county", "employment"],
                answer_schema=["county", "total_employment"],
                exhaustive=False,
                use_retrieval_support=False,
                route_notes=route_notes + ["county_employment_ranking"],
            )

        if ("highest employment" in normalized_question or "largest employment" in normalized_question) and "county" in normalized_question:
            return StructuredQueryPlan(
                route_name="top_company_by_employment",
                question_type="top_company_by_employment",
                filters=filters,
                supplier_only=supplier_only,
                requires_structured_context=True,
                group_by=[],
                aggregate=None,
                aggregate_field=None,
                sort_by="employment",
                sort_desc=True,
                top_n=1,
                requested_fields=["company", "employment", "ev_supply_chain_role", "county"],
                dedupe_by="company",
                answer_schema=["company", "employment", "ev_supply_chain_role", "county"],
                exhaustive=False,
                use_retrieval_support=False,
                route_notes=route_notes + ["top_company_by_employment"],
            )

        if ("employment" in normalized_question) and (
            normalized_question.startswith("top ")
            or "which three" in normalized_question
            or "largest employment" in normalized_question
        ):
            return StructuredQueryPlan(
                route_name="employment_ranked_companies",
                question_type="employment_ranked_companies",
                filters=filters,
                supplier_only=supplier_only,
                requires_structured_context=True,
                group_by=[],
                aggregate=None,
                aggregate_field=None,
                sort_by="employment",
                sort_desc=True,
                top_n=self._extract_top_n(question, self.top_n_default),
                requested_fields=["company", "employment", "ev_supply_chain_role", "category", "county", "product_service"],
                dedupe_by="company",
                ontology_buckets=ontology_buckets,
                ontology_match_mode=self.ontology_match_mode,
                preferred_match_fields=["ev_supply_chain_role", "product_service"] if ontology_buckets else [],
                answer_schema=["company", "employment", "ev_supply_chain_role", "category", "county"],
                exhaustive=False,
                use_retrieval_support=False,
                route_notes=route_notes + ["employment_ranked_companies"],
            )

        if "what locations does" in normalized_question and "facility" in normalized_question:
            return StructuredQueryPlan(
                route_name="company_locations",
                question_type="company_locations",
                filters=filters,
                supplier_only=False,
                requires_structured_context=True,
                group_by=["updated_location", "primary_facility_type"],
                aggregate="count",
                aggregate_field="record_id",
                sort_by="updated_location_normalized",
                sort_desc=False,
                top_n=None,
                requested_fields=["updated_location", "primary_facility_type", "count"],
                target_company=target_company,
                answer_schema=["updated_location", "primary_facility_type"],
                exhaustive=True,
                use_retrieval_support=False,
                route_notes=route_notes + ["company_locations"],
            )

        if self._mentions_rd(question):
            return StructuredQueryPlan(
                route_name="rd_facilities",
                question_type="rd_facilities",
                filters=filters,
                supplier_only=False,
                requires_structured_context=True,
                group_by=[],
                aggregate=None,
                aggregate_field=None,
                sort_by="updated_location_normalized",
                sort_desc=False,
                top_n=None,
                requested_fields=["updated_location", "county", "company", "primary_facility_type", "product_service"],
                dedupe_by="company",
                answer_schema=["updated_location", "company", "primary_facility_type", "product_service"],
                exhaustive=True,
                use_retrieval_support=True,
                retrieval_query="Georgia R&D automotive facilities",
                route_notes=route_notes + ["rd_facilities"],
            )

        if normalized_question.startswith(("how many", "count")) and ontology_buckets:
            preferred_fields = ["product_service", "ev_supply_chain_role"]
            if "lithium ion" in normalized_question and ("cells" in normalized_question or "electrolytes" in normalized_question):
                route_notes.append("strict_count_profile=lithium_ion_core")
            return StructuredQueryPlan(
                route_name="count_ontology_companies",
                question_type="count_ontology_companies",
                filters=filters,
                supplier_only=False,
                requires_structured_context=True,
                group_by=[],
                aggregate="count",
                aggregate_field="company",
                sort_by="company_normalized",
                sort_desc=False,
                top_n=None,
                requested_fields=["company", "category", "ev_supply_chain_role", "product_service"],
                dedupe_by="company",
                ontology_buckets=ontology_buckets,
                ontology_match_mode=self.ontology_match_mode,
                preferred_match_fields=preferred_fields,
                answer_schema=["count", "companies", "category", "product_service"],
                exhaustive=True,
                use_retrieval_support=False,
                route_notes=route_notes + [f"ontology={','.join(ontology_buckets)}"],
            )

        if ontology_buckets and normalized_question.startswith(("show", "list", "which", "identify", "find", "map", "top")):
            preferred_fields: list[str] = []
            if "supply chain role" in normalized_question:
                preferred_fields = ["ev_supply_chain_role"]
            elif "product" in normalized_question or "materials" in normalized_question:
                preferred_fields = ["product_service", "ev_supply_chain_role"]
            requested_fields = ["company", "category", "ev_supply_chain_role", "product_service", "primary_oems", "employment", "updated_location", "primary_facility_type"]
            if "employment" in normalized_question and "highest" in normalized_question:
                requested_fields = ["company", "employment", "ev_supply_chain_role", "county"]
            return StructuredQueryPlan(
                route_name="ontology_filtered_list",
                question_type="ontology_filtered_list",
                filters=filters,
                supplier_only=supplier_only,
                requires_structured_context=True,
                group_by=[],
                aggregate=None,
                aggregate_field=None,
                sort_by="company_normalized",
                sort_desc=False,
                top_n=self.top_n_default if normalized_question.startswith("top ") else None,
                requested_fields=requested_fields,
                dedupe_by="company",
                ontology_buckets=ontology_buckets,
                ontology_match_mode=self.ontology_match_mode,
                preferred_match_fields=preferred_fields,
                answer_schema=requested_fields,
                exhaustive=not normalized_question.startswith("top "),
                use_retrieval_support=True,
                retrieval_query=" ".join([question] + self.ontology.retrieval_terms(ontology_buckets)),
                route_notes=route_notes + [f"ontology={','.join(ontology_buckets)}"],
            )

        if "show which primary oems" in normalized_question or "primary oems" in normalized_question:
            return StructuredQueryPlan(
                route_name="oem_mapping",
                question_type="oem_mapping",
                filters=filters,
                supplier_only=supplier_only,
                requires_structured_context=True,
                group_by=[],
                aggregate=None,
                aggregate_field=None,
                sort_by="company_normalized",
                sort_desc=False,
                top_n=None,
                requested_fields=["company", "primary_oems", "ev_supply_chain_role", "updated_location", "primary_facility_type", "employment", "product_service"],
                dedupe_by="company",
                answer_schema=["company", "primary_oems", "ev_supply_chain_role"],
                exhaustive=True,
                use_retrieval_support=True,
                route_notes=route_notes + ["oem_mapping"],
            )

        if normalized_question.startswith(("how many", "count")):
            return StructuredQueryPlan(
                route_name="count_filtered_rows",
                question_type="count_filtered_rows",
                filters=filters,
                supplier_only=supplier_only,
                requires_structured_context=True,
                group_by=[],
                aggregate="count",
                aggregate_field="record_id",
                sort_by=None,
                sort_desc=True,
                top_n=None,
                requested_fields=["count"],
                answer_schema=["count"],
                exhaustive=False,
                use_retrieval_support=False,
                route_notes=route_notes + ["count_filtered_rows"],
            )

        if normalized_question.startswith(("show", "list", "which", "identify", "name", "map", "find", "top")):
            return StructuredQueryPlan(
                route_name="filtered_list",
                question_type="filtered_list",
                filters=filters,
                supplier_only=supplier_only,
                requires_structured_context=True,
                group_by=[],
                aggregate=None,
                aggregate_field=None,
                sort_by="company_normalized",
                sort_desc=False,
                top_n=self.top_n_default if normalized_question.startswith("top ") else None,
                requested_fields=["company", "category", "ev_supply_chain_role", "product_service", "primary_oems", "employment", "updated_location", "primary_facility_type"],
                dedupe_by="company" if "company" in normalized_question or "companies" in normalized_question or "suppliers" in normalized_question else None,
                answer_schema=["company", "category", "ev_supply_chain_role", "product_service", "primary_oems"],
                exhaustive=not normalized_question.startswith("top "),
                use_retrieval_support=True,
                route_notes=route_notes + ["filtered_list"],
            )

        return StructuredQueryPlan(
            route_name="lookup",
            question_type="lookup",
            filters=filters,
            supplier_only=supplier_only,
            requires_structured_context=False,
            group_by=[],
            aggregate=None,
            aggregate_field=None,
            sort_by=None,
            sort_desc=False,
            top_n=None,
            requested_fields=["company", "category", "ev_supply_chain_role", "product_service", "primary_oems", "employment"],
            target_company=target_company,
            answer_schema=["company", "category", "ev_supply_chain_role", "product_service", "primary_oems"],
            exhaustive=False,
            use_retrieval_support=True,
            route_notes=route_notes + ["lookup"],
        )
