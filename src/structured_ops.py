from __future__ import annotations

from typing import Any

import pandas as pd

from src.ontology import DomainOntology
from src.query_router import QueryRouter, StructuredQueryPlan
from src.schemas import StructuredOpArtifact


SUPPLIER_FILTER_DESCRIPTION = (
    "is_supplier == True OR classification_method == Supplier OR supplier_or_affiliation_type contains "
    "'Automotive supply chain participant'"
)


class StructuredOpsEngine:
    def __init__(
        self,
        kb_frame: pd.DataFrame,
        ontology: DomainOntology,
        top_n_default: int = 5,
        max_evidence_rows: int = 25,
        ontology_match_mode: str = "any",
    ) -> None:
        self.kb_frame = kb_frame.copy()
        self.ontology = ontology
        self.top_n_default = top_n_default
        self.max_evidence_rows = max_evidence_rows
        self.ontology_match_mode = ontology_match_mode
        self.router = QueryRouter(
            kb_frame=self.kb_frame,
            ontology=self.ontology,
            top_n_default=top_n_default,
            ontology_match_mode=ontology_match_mode,
        )

    def plan_question(self, question: str) -> StructuredQueryPlan:
        return self.router.route(question)

    @staticmethod
    def _apply_single_filter(frame: pd.DataFrame, filter_spec: dict[str, Any]) -> pd.DataFrame:
        field = filter_spec["field"]
        op = filter_spec["op"]
        value = filter_spec["value"]
        if field not in frame.columns:
            return frame

        series = frame[field]
        if op == "eq":
            return frame[series.astype(str).str.lower() == str(value).lower()]
        if op == "contains":
            return frame[series.astype(str).str.lower().str.contains(str(value).lower(), na=False)]
        if op == "in":
            allowed = {str(item).lower() for item in value}
            return frame[series.astype(str).str.lower().isin(allowed)]
        if op == "gt":
            return frame[pd.to_numeric(series, errors="coerce").fillna(0.0) > float(value)]
        if op == "gte":
            return frame[pd.to_numeric(series, errors="coerce").fillna(0.0) >= float(value)]
        if op == "lt":
            return frame[pd.to_numeric(series, errors="coerce").fillna(0.0) < float(value)]
        if op == "lte":
            return frame[pd.to_numeric(series, errors="coerce").fillna(0.0) <= float(value)]
        return frame

    def _apply_filters(self, frame: pd.DataFrame, plan: StructuredQueryPlan) -> pd.DataFrame:
        working = frame.copy()
        for filter_spec in plan.filters:
            working = self._apply_single_filter(working, filter_spec)
        if plan.supplier_only and "is_supplier" in working.columns:
            working = working[working["is_supplier"]]
        return working

    def _match_ontology_row(self, row: dict[str, Any], plan: StructuredQueryPlan) -> list[str]:
        if not plan.ontology_buckets:
            return []
        hits: list[str] = []
        for bucket_name in plan.ontology_buckets:
            matched = self.ontology.row_matches_bucket(
                row=row,
                bucket_name=bucket_name,
                preferred_fields=plan.preferred_match_fields or None,
            )
            if matched:
                hits.append(bucket_name)
        return hits

    def _apply_ontology_filters(self, frame: pd.DataFrame, plan: StructuredQueryPlan) -> pd.DataFrame:
        if not plan.ontology_buckets:
            return frame

        rows: list[dict[str, Any]] = []
        for _, row in frame.iterrows():
            payload = row.to_dict()
            hits = self._match_ontology_row(payload, plan)
            if not hits:
                continue
            payload["matched_ontology_buckets"] = hits
            rows.append(payload)

        if not rows:
            empty = frame.iloc[0:0].copy()
            empty["matched_ontology_buckets"] = [[] for _ in range(len(empty))]
            return empty

        filtered = pd.DataFrame(rows)
        if plan.ontology_match_mode == "all":
            filtered = filtered[
                filtered["matched_ontology_buckets"].apply(
                    lambda item: all(bucket in item for bucket in plan.ontology_buckets)
                )
            ]
        return filtered

    @staticmethod
    def _apply_route_specific_refinement(frame: pd.DataFrame, plan: StructuredQueryPlan) -> pd.DataFrame:
        if "strict_count_profile=lithium_ion_core" in plan.route_notes:
            required_terms = (
                "lithium ion battery recycler",
                "lithium ion battery materials",
                "battery electrolyte",
                "battery cells for electric mobility",
            )
            return frame[
                frame["product_service_normalized"].astype(str).apply(
                    lambda value: any(term in value for term in required_terms)
                )
            ]
        return frame

    @staticmethod
    def _render_rows(rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "No matching structured rows found."
        rendered: list[str] = []
        for index, row in enumerate(rows, start=1):
            payload = " | ".join(f"{key}={value}" for key, value in row.items())
            rendered.append(f"{index}. {payload}")
        return "\n".join(rendered)

    def _build_artifact(
        self,
        name: str,
        plan: StructuredQueryPlan,
        rows: list[dict[str, Any]],
        plan_summary: str,
    ) -> StructuredOpArtifact:
        filters = list(plan.filters)
        if plan.supplier_only:
            filters.append({"field": "supplier_only", "op": "custom", "value": SUPPLIER_FILTER_DESCRIPTION})
        if plan.ontology_buckets:
            filters.append({"field": "ontology_buckets", "op": plan.ontology_match_mode, "value": plan.ontology_buckets})
        rendered_text = (
            f"STRUCTURED EVIDENCE: {name}\n"
            f"ROUTE: {plan.route_name}\n"
            f"ANSWER FIELDS: {', '.join(plan.answer_schema or plan.requested_fields)}\n"
            f"SUMMARY: {plan_summary}\n"
            f"RESULTS:\n{self._render_rows(rows)}"
        )
        return StructuredOpArtifact(
            name=name,
            plan_summary=plan_summary,
            filters=filters,
            result_rows=rows,
            rendered_text=rendered_text,
        )

    def _project_rows(self, frame: pd.DataFrame, plan: StructuredQueryPlan) -> list[dict[str, Any]]:
        working = frame.copy()
        if plan.dedupe_by and plan.dedupe_by in working.columns:
            working = working.drop_duplicates(subset=[plan.dedupe_by], keep="first")
        if plan.sort_by and plan.sort_by in working.columns:
            working = working.sort_values(plan.sort_by, ascending=not plan.sort_desc)
        if plan.top_n is not None:
            working = working.head(plan.top_n)
        if self.max_evidence_rows is not None:
            working = working.head(self.max_evidence_rows)
        fields = [field for field in plan.requested_fields if field in working.columns]
        return working[fields].to_dict(orient="records") if fields else []

    def _execute_count_ontology_companies(
        self,
        filtered: pd.DataFrame,
        plan: StructuredQueryPlan,
    ) -> StructuredOpArtifact:
        working = filtered.copy()
        if "company" in working.columns:
            working = working.drop_duplicates(subset=["company"], keep="first")
        working = working.sort_values("company_normalized", ascending=True)
        detail_rows = self._project_rows(working, plan)
        rows = [{"matching_company_count": int(len(working))}] + detail_rows
        return self._build_artifact(
            name="count_ontology_companies",
            plan=plan,
            rows=rows,
            plan_summary="Counted unique companies that match the routed ontology buckets.",
        )

    def _execute_count_filtered_rows(
        self,
        filtered: pd.DataFrame,
        plan: StructuredQueryPlan,
    ) -> StructuredOpArtifact:
        return self._build_artifact(
            name="count_filtered_rows",
            plan=plan,
            rows=[{"count": int(len(filtered))}],
            plan_summary="Counted filtered rows deterministically.",
        )

    def _execute_county_employment_ranking(
        self,
        filtered: pd.DataFrame,
        plan: StructuredQueryPlan,
    ) -> StructuredOpArtifact:
        grouped = (
            filtered[filtered["county"].astype(str) != ""]
            .groupby("county", dropna=False)["employment"]
            .sum()
            .reset_index()
            .rename(columns={"employment": "total_employment"})
            .sort_values("total_employment", ascending=False)
        )
        if plan.top_n is not None:
            grouped = grouped.head(plan.top_n)
        rows = grouped.to_dict(orient="records")
        return self._build_artifact(
            name="county_employment_totals",
            plan=plan,
            rows=rows,
            plan_summary="Grouped matching rows by county and summed employment deterministically.",
        )

    def _execute_company_locations(
        self,
        filtered: pd.DataFrame,
        plan: StructuredQueryPlan,
    ) -> StructuredOpArtifact:
        grouped = (
            filtered.groupby(["updated_location", "primary_facility_type"], dropna=False)
            .agg(count=("record_id", "count"))
            .reset_index()
            .sort_values(["updated_location", "primary_facility_type"], ascending=True)
        )
        rows = grouped.head(self.max_evidence_rows).to_dict(orient="records")
        return self._build_artifact(
            name="company_location_facility_map",
            plan=plan,
            rows=rows,
            plan_summary="Grouped matching company rows by location and facility type.",
        )

    def _execute_sole_source_roles(
        self,
        filtered: pd.DataFrame,
        plan: StructuredQueryPlan,
    ) -> StructuredOpArtifact:
        rows: list[dict[str, Any]] = []
        working = filtered.copy()
        working["sole_source_role_key"] = working.apply(
            lambda row: (
                str(row.get("ev_supply_chain_role", "")).strip()
                or (
                    str(row.get("primary_facility_type", "")).strip()
                    if str(row.get("primary_facility_type", "")).strip().startswith("EV ")
                    else ""
                )
            ),
            axis=1,
        )
        working = working[working["sole_source_role_key"].astype(str) != ""]
        grouped = working.groupby("sole_source_role_key", dropna=False)["company"].agg(lambda values: sorted(set(values)))
        for role, companies in grouped.items():
            if len(companies) == 1:
                rows.append({"ev_supply_chain_role": role, "company": companies[0]})
        rows = sorted(rows, key=lambda item: str(item["ev_supply_chain_role"]).lower())
        return self._build_artifact(
            name="sole_source_roles",
            plan=plan,
            rows=rows,
            plan_summary="Found EV supply-chain roles served by exactly one unique company.",
        )

    def _execute_oem_supplier_network(
        self,
        filtered: pd.DataFrame,
        plan: StructuredQueryPlan,
    ) -> StructuredOpArtifact:
        rows = self._project_rows(filtered, plan)
        return self._build_artifact(
            name="oem_supplier_network",
            plan=plan,
            rows=rows,
            plan_summary=f"Mapped supplier records linked to target OEM '{plan.target_oem}'.",
        )

    def _execute_rd_facilities(
        self,
        filtered: pd.DataFrame,
        plan: StructuredQueryPlan,
    ) -> StructuredOpArtifact:
        working = filtered[filtered["is_rd_facility"]].copy() if "is_rd_facility" in filtered.columns else filtered
        rows = self._project_rows(working, plan)
        return self._build_artifact(
            name="rd_facilities",
            plan=plan,
            rows=rows,
            plan_summary="Selected automotive records with R&D facility signals.",
        )

    def _execute_generic_rows(
        self,
        filtered: pd.DataFrame,
        plan: StructuredQueryPlan,
    ) -> StructuredOpArtifact:
        artifact_name = plan.question_type
        if plan.question_type == "oem_mapping":
            artifact_name = "oem_mapping"
        rows = self._project_rows(filtered, plan)
        return self._build_artifact(
            name=artifact_name,
            plan=plan,
            rows=rows,
            plan_summary=f"Rendered filtered rows for route '{plan.route_name}'.",
        )

    def execute(self, question: str) -> tuple[StructuredQueryPlan, list[StructuredOpArtifact]]:
        plan = self.plan_question(question)
        if not plan.requires_structured_context:
            return plan, []

        filtered = self._apply_filters(self.kb_frame, plan)
        filtered = self._apply_ontology_filters(filtered, plan)
        filtered = self._apply_route_specific_refinement(filtered, plan)
        support_ids: list[str] = []
        for record_id in filtered.get("record_id", pd.Series(dtype=str)).astype(str).tolist():
            if record_id and record_id not in support_ids:
                support_ids.append(record_id)
        plan.support_record_ids = support_ids

        if plan.question_type == "county_employment_ranking":
            return plan, [self._execute_county_employment_ranking(filtered, plan)]
        if plan.question_type == "company_locations":
            return plan, [self._execute_company_locations(filtered, plan)]
        if plan.question_type == "count_ontology_companies":
            return plan, [self._execute_count_ontology_companies(filtered, plan)]
        if plan.question_type == "count_filtered_rows":
            return plan, [self._execute_count_filtered_rows(filtered, plan)]
        if plan.question_type == "sole_source_roles":
            return plan, [self._execute_sole_source_roles(filtered, plan)]
        if plan.question_type == "oem_supplier_network":
            return plan, [self._execute_oem_supplier_network(filtered, plan)]
        if plan.question_type == "rd_facilities":
            return plan, [self._execute_rd_facilities(filtered, plan)]
        return plan, [self._execute_generic_rows(filtered, plan)]
