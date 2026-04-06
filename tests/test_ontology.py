from __future__ import annotations


def _row(kb_frame, company: str) -> dict:
    return kb_frame[kb_frame["company"] == company].iloc[0].to_dict()


def test_ontology_matches_battery_materials_without_overmatching_cells(kb_frame, ontology) -> None:
    duckyang = _row(kb_frame, "Duckyang")
    hitachi = _row(kb_frame, "Hitachi Astemo Americas Inc.")

    assert ontology.row_matches_bucket(duckyang, "battery_materials")
    assert not ontology.row_matches_bucket(hitachi, "battery_materials")
    assert ontology.row_matches_bucket(hitachi, "battery_cells")


def test_ontology_matches_battery_parts_and_wiring_harness_buckets(kb_frame, ontology) -> None:
    hyundai_motor_group = _row(kb_frame, "Hyundai Motor Group")
    wika = _row(kb_frame, "WIKA USA")
    woory = _row(kb_frame, "Woory Industrial Co.")

    assert ontology.row_matches_bucket(hyundai_motor_group, "battery_parts")
    assert ontology.row_matches_bucket(wika, "wiring_harness")
    assert not ontology.row_matches_bucket(woory, "wiring_harness")
