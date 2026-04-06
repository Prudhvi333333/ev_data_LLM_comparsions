from __future__ import annotations

from src.query_router import StructuredQueryPlan
from src.schemas import StructuredOpArtifact


def render_structured_answer(
    plan: StructuredQueryPlan,
    artifact: StructuredOpArtifact | None,
    not_found_text: str,
) -> str | None:
    if artifact is None or not artifact.result_rows:
        return None

    rows = artifact.result_rows

    if plan.route_name == "oem_supplier_network":
        return "\n".join(
            f"{row['company']} | {row['category']} | Role: {row['ev_supply_chain_role']} | OEMs: {row['primary_oems']}"
            for row in rows
        )

    if plan.route_name == "count_ontology_companies":
        count_row = rows[0]
        detail_rows = rows[1:]
        lines = [str(count_row.get("matching_company_count", 0))]
        lines.extend(
            f"{row['company']} | {row['category']} | {row['product_service']}"
            for row in detail_rows
        )
        return "\n".join(lines)

    if plan.route_name == "sole_source_roles":
        return "\n".join(
            f"{row['ev_supply_chain_role']} | {row['company']}"
            for row in rows
        )

    if plan.route_name == "rd_facilities":
        return "\n".join(
            f"{row['updated_location']} | {row['company']} | {row['product_service']}"
            for row in rows
        )

    return None
