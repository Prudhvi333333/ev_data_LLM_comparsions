from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from src.utils.logger import get_logger


logger = get_logger("sample_data")


KB_COLUMNS = [
    "Company",
    "Category",
    "Industry Group",
    "Location",
    "Updated Location",
    "Address",
    "Latitude",
    "Longitude",
    "Primary Facility Type",
    "EV Supply Chain Role",
    "Primary OEMs",
    "Supplier or Affiliation Type",
    "Employment",
    "Product / Service",
    "EV / Battery Relevant",
    "Classification Method",
]


QUESTION_COLUMNS = [
    "Num",
    "Use Case Category",
    "Question",
    "Human Validated Answers",
    "Answer from Web",
]


def _get_config_value(config: SimpleNamespace | dict, dotted_path: str, default: str = "") -> str:
    current = config
    for key in dotted_path.split("."):
        if isinstance(current, dict):
            current = current.get(key)
        else:
            current = getattr(current, key, None)
        if current is None:
            return default
    return str(current)


def _build_kb_frame() -> pd.DataFrame:
    tiers = ["Tier 1", "Tier 2", "Tier 2/3", "Tier 3"]
    roles = [
        "Battery Cell Manufacturing",
        "Battery Pack Assembly",
        "Thermal Management Systems",
        "Raw Material Processing",
        "Battery Recycling",
        "General Automotive Components",
    ]
    oems = ["Hyundai", "Kia", "Rivian", "Ford", "GM", "SK On"]
    cities = ["Atlanta", "Savannah", "Augusta", "Macon", "Columbus", "Dalton"]
    groups = [
        "Battery Manufacturing",
        "Thermal Systems",
        "Metal Processing",
        "Power Electronics",
        "Logistics",
        "Vehicle Assembly",
    ]
    facilities = [
        "Manufacturing Plant",
        "Assembly Facility",
        "R&D Center",
        "Distribution Hub",
        "Recycling Plant",
    ]
    affiliations = ["Independent Supplier", "Joint Venture", "OEM Subsidiary", "Strategic Partner"]

    rows = []
    for idx in range(205):
        tier = tiers[idx % len(tiers)]
        role = roles[idx % len(roles)]
        city = cities[idx % len(cities)]
        oem_main = oems[idx % len(oems)]
        oem_partner = oems[(idx + 2) % len(oems)]
        rows.append(
            {
                "Company": f"GNEM Company {idx + 1:03d}",
                "Category": tier,
                "Industry Group": groups[idx % len(groups)],
                "Location": city,
                "Updated Location": f"{city}, Georgia",
                "Address": f"{100 + idx} Innovation Parkway, {city}, GA",
                "Latitude": round(32.0 + (idx % 50) * 0.05, 6),
                "Longitude": round(-85.0 - (idx % 50) * 0.05, 6),
                "Primary Facility Type": facilities[idx % len(facilities)],
                "EV Supply Chain Role": role,
                "Primary OEMs": f"{oem_main}; {oem_partner}",
                "Supplier or Affiliation Type": affiliations[idx % len(affiliations)],
                "Employment": 80 + (idx % 40) * 25,
                "Product / Service": f"{role} components and services for EV programs {idx + 1:03d}",
                "EV / Battery Relevant": "Yes" if "Battery" in role or "Thermal" in role else "Partial",
                "Classification Method": "Synthetic baseline record",
            }
        )
    return pd.DataFrame(rows, columns=KB_COLUMNS)


def _build_questions_frame(kb_df: pd.DataFrame) -> pd.DataFrame:
    question_templates = [
        (
            "Supply Chain Mapping & Visibility",
            "List all Tier 1 battery cell companies serving Hyundai in Georgia.",
            lambda frame: ", ".join(
                frame[
                    (frame["Category"] == "Tier 1")
                    & frame["EV Supply Chain Role"].str.contains("Battery Cell", case=False, na=False)
                    & frame["Primary OEMs"].str.contains("Hyundai", case=False, na=False)
                ]["Company"].head(12).tolist()
            ),
        ),
        (
            "EV Battery Role Analysis",
            "How many battery cell companies are in Georgia?",
            lambda frame: str(
                frame["EV Supply Chain Role"].str.contains("Battery Cell", case=False, na=False).sum()
            ),
        ),
        (
            "OEM-Supplier Relationships",
            "Which companies supply thermal management systems to Kia?",
            lambda frame: ", ".join(
                frame[
                    frame["EV Supply Chain Role"].str.contains("Thermal Management", case=False, na=False)
                    & frame["Primary OEMs"].str.contains("Kia", case=False, na=False)
                ]["Company"].head(12).tolist()
            ),
        ),
        (
            "Geographic Analysis",
            "Which EV suppliers are located in Savannah?",
            lambda frame: ", ".join(frame[frame["Location"] == "Savannah"]["Company"].head(12).tolist()),
        ),
        (
            "Employment & Facility Intelligence",
            "Compare Tier 1 and Tier 2 suppliers in Atlanta.",
            lambda frame: "; ".join(
                frame[
                    frame["Category"].isin(["Tier 1", "Tier 2"])
                    & (frame["Location"] == "Atlanta")
                ][["Company", "Category"]]
                .head(12)
                .apply(lambda row: f"{row['Company']} ({row['Category']})", axis=1)
                .tolist()
            ),
        ),
    ]

    rows = []
    for idx in range(50):
        category, question, answer_fn = question_templates[idx % len(question_templates)]
        suffix = "" if idx < 5 else f" Focus on batch {idx + 1}."
        rows.append(
            {
                "Num": idx + 1,
                "Use Case Category": category,
                "Question": question + suffix,
                "Human Validated Answers": answer_fn(kb_df),
                "Answer from Web": "",
            }
        )
    return pd.DataFrame(rows, columns=QUESTION_COLUMNS)


def ensure_sample_data(config: SimpleNamespace | dict) -> None:
    kb_path = Path(_get_config_value(config, "paths.kb", "data/kb/gnem_auto_landscape.xlsx"))
    questions_path = Path(
        _get_config_value(config, "paths.questions", "data/questions/questions_50.xlsx")
    )

    kb_path.parent.mkdir(parents=True, exist_ok=True)
    questions_path.parent.mkdir(parents=True, exist_ok=True)

    if kb_path.exists() and questions_path.exists():
        return

    kb_df = _build_kb_frame()
    questions_df = _build_questions_frame(kb_df)

    if not kb_path.exists():
        kb_df.to_excel(kb_path, index=False)
        logger.info("Created synthetic KB sample data at %s", kb_path)

    if not questions_path.exists():
        questions_df.to_excel(questions_path, index=False)
        logger.info("Created synthetic question sample data at %s", questions_path)
