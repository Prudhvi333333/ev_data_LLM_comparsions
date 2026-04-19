from __future__ import annotations

from src.structured_ops import StructuredOpsEngine


def _engine(kb_frame, ontology) -> StructuredOpsEngine:
    return StructuredOpsEngine(
        kb_frame=kb_frame,
        ontology=ontology,
        top_n_default=10,
        max_evidence_rows=50,
        ontology_match_mode="any",
    )


def test_q8_county_employment_uses_strict_tier1_rows(kb_frame, ontology) -> None:
    engine = _engine(kb_frame, ontology)
    question = "Which county have the highest total Employment among Tier 1 suppliers only?"
    plan, artifacts = engine.execute(question)

    assert plan.question_type == "county_employment_ranking"
    assert artifacts
    first_row = artifacts[0].result_rows[0]
    assert first_row["county"] == "Troup County"
    assert first_row["total_employment"] == 2435.0


def test_q13_oem_supplier_network_uses_primary_oem_link_not_oem_company(kb_frame, ontology) -> None:
    engine = _engine(kb_frame, ontology)
    question = "Show the full supplier network linked to Rivian Automotive in Georgia, broken down by tier and EV Supply Chain Role."
    plan, artifacts = engine.execute(question)

    rows = artifacts[0].result_rows
    companies = {row["company"] for row in rows}
    assert plan.question_type == "oem_supplier_network"
    assert {"Duckyang", "Enchem America Inc.", "GSC Steel Stamping LLC", "Suzuki Manufacturing of America Corp."} <= companies


def test_q15_wiring_harness_route_prefers_role_and_product_ontology(kb_frame, ontology) -> None:
    engine = _engine(kb_frame, ontology)
    question = "Identify all Georgia companies with an EV Supply Chain Role related to wiring harnesses and show their Primary OEMs."
    _, artifacts = engine.execute(question)

    companies = {row["company"] for row in artifacts[0].result_rows}
    assert {"WIKA USA", "Woodbridge Foam Corp."} <= companies
    assert "Woory Industrial Co." not in companies


def test_q17_harness_and_electrical_distribution_route_stays_on_target(kb_frame, ontology) -> None:
    engine = _engine(kb_frame, ontology)
    question = "Which Georgia companies manufacture high-voltage wiring harnesses or EV electrical distribution components suitable for BEV platforms?"
    _, artifacts = engine.execute(question)

    companies = {row["company"] for row in artifacts[0].result_rows}
    assert {"WIKA USA", "Woodbridge Foam Corp."} <= companies
    assert "Woory Industrial Co." not in companies


def test_q35_count_uses_strict_battery_materials_cells_and_electrolyte_matcher(kb_frame, ontology) -> None:
    engine = _engine(kb_frame, ontology)
    question = "How many Georgia companies are now producing lithium-ion battery materials, cells, or electrolytes?"
    plan, artifacts = engine.execute(question)

    rows = artifacts[0].result_rows
    companies = {row["company"] for row in rows[1:]}
    assert plan.question_type == "count_ontology_companies"
    assert rows[0]["matching_company_count"] == 5
    assert companies == {
        "F&P Georgia Manufacturing",
        "Hitachi Astemo Americas Inc.",
        "Hollingsworth & Vose Co.",
        "Honda Development & Manufacturing",
        "IMMI",
    }


def test_q24_battery_parts_and_enclosures_match_expected_tier_1_2_companies(kb_frame, ontology) -> None:
    engine = _engine(kb_frame, ontology)
    question = "Which Georgia companies manufacture battery parts or enclosure systems and are classified as Tier 1/2, making them ready for direct OEM engagement and show which Primary OEMs they are linked to."
    _, artifacts = engine.execute(question)

    companies = {row["company"] for row in artifacts[0].result_rows}
    assert companies == {
        "F&P Georgia Manufacturing",
        "Hitachi Astemo Americas Inc.",
        "Hollingsworth & Vose Co.",
        "Honda Development & Manufacturing",
        "Hyundai Motor Group",
        "IMMI",
    }


def test_q27_sole_source_roles_includes_ev_facility_fallback_role(kb_frame, ontology) -> None:
    engine = _engine(kb_frame, ontology)
    question = "Which EV Supply Chain Roles in Georgia are served by only a single company, creating a single-point-of-failure risk for the state's EV ecosystem?"
    _, artifacts = engine.execute(question)

    rows = artifacts[0].result_rows
    roles = {row["ev_supply_chain_role"] for row in rows}
    assert "EV thermal systems and electronics" in roles
    assert "Tier 1 automotive components" in roles
    assert "Vehicle safety systems OEM (EV + ICE)" in roles
