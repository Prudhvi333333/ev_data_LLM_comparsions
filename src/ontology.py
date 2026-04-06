from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
import re

import pandas as pd
import yaml
from pydantic import BaseModel, Field


_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


def normalize_for_match(value: Any) -> str:
    raw = str(value or "").strip().lower()
    raw = raw.replace("‑", "-").replace("–", "-").replace("—", "-")
    raw = raw.replace("&", " and ")
    normalized = _NORMALIZE_RE.sub(" ", raw)
    return re.sub(r"\s+", " ", normalized).strip()


class OntologyBucket(BaseModel):
    aliases: list[str] = Field(default_factory=list)
    synonyms: list[str] = Field(default_factory=list)
    exact_phrases: list[str] = Field(default_factory=list)
    exclusions: list[str] = Field(default_factory=list)
    field_targets: list[str] = Field(default_factory=lambda: ["ev_supply_chain_role", "product_service"])
    question_hints: list[str] = Field(default_factory=list)
    retrieval_expansion_terms: list[str] = Field(default_factory=list)

    @property
    def positive_terms(self) -> list[str]:
        ordered = self.exact_phrases + self.aliases + self.synonyms
        unique: list[str] = []
        for term in ordered:
            normalized = normalize_for_match(term)
            if normalized and normalized not in unique:
                unique.append(normalized)
        return unique

    @property
    def normalized_exclusions(self) -> list[str]:
        return [normalize_for_match(term) for term in self.exclusions if normalize_for_match(term)]

    @property
    def normalized_question_hints(self) -> list[str]:
        ordered = self.question_hints or (self.aliases + self.synonyms)
        unique: list[str] = []
        for term in ordered:
            normalized = normalize_for_match(term)
            if normalized and normalized not in unique:
                unique.append(normalized)
        return unique


class OntologyConfig(BaseModel):
    version: str = "v3"
    buckets: dict[str, OntologyBucket]


@dataclass
class OntologyRowMatch:
    bucket_name: str
    matched: bool
    matched_fields: dict[str, list[str]] = field(default_factory=dict)


class DomainOntology:
    def __init__(self, config: OntologyConfig) -> None:
        self.config = config
        self.version = config.version

    @classmethod
    def load(cls, path: str | Path) -> "DomainOntology":
        ontology_path = Path(path)
        payload = yaml.safe_load(ontology_path.read_text(encoding="utf-8")) or {}
        config = OntologyConfig.model_validate(payload)
        return cls(config=config)

    def bucket_names(self) -> list[str]:
        return list(self.config.buckets)

    def question_buckets(self, question: str) -> list[str]:
        normalized_question = normalize_for_match(question)
        matched: list[str] = []
        for bucket_name, bucket in self.config.buckets.items():
            if any(term in normalized_question for term in bucket.normalized_question_hints):
                matched.append(bucket_name)
        return matched

    def row_match_details(
        self,
        row: Mapping[str, Any],
        bucket_name: str,
        preferred_fields: list[str] | None = None,
    ) -> OntologyRowMatch:
        bucket = self.config.buckets[bucket_name]
        preferred = [field for field in (preferred_fields or []) if field in bucket.field_targets]
        search_fields = preferred or bucket.field_targets
        matches: dict[str, list[str]] = {}

        for field in search_fields:
            normalized_value = normalize_for_match(row.get(field, ""))
            if not normalized_value:
                continue
            if any(exclusion in normalized_value for exclusion in bucket.normalized_exclusions):
                continue
            hits = [term for term in bucket.positive_terms if term and term in normalized_value]
            if hits:
                matches[field] = hits
        return OntologyRowMatch(bucket_name=bucket_name, matched=bool(matches), matched_fields=matches)

    def row_matches_bucket(
        self,
        row: Mapping[str, Any],
        bucket_name: str,
        preferred_fields: list[str] | None = None,
    ) -> bool:
        return self.row_match_details(row=row, bucket_name=bucket_name, preferred_fields=preferred_fields).matched

    def annotate_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        annotated = frame.copy()
        tag_values: list[list[str]] = []
        tag_text_values: list[str] = []
        for _, row in annotated.iterrows():
            row_tags = [
                bucket_name
                for bucket_name in self.bucket_names()
                if self.row_matches_bucket(row.to_dict(), bucket_name=bucket_name)
            ]
            tag_values.append(row_tags)
            tag_text_values.append(", ".join(row_tags))
        annotated["ontology_tags"] = tag_values
        annotated["ontology_tags_text"] = tag_text_values
        return annotated

    def retrieval_terms(self, bucket_names: list[str]) -> list[str]:
        terms: list[str] = []
        for bucket_name in bucket_names:
            bucket = self.config.buckets.get(bucket_name)
            if bucket is None:
                continue
            for term in bucket.retrieval_expansion_terms + bucket.aliases + bucket.synonyms:
                normalized = term.strip()
                if normalized and normalized not in terms:
                    terms.append(normalized)
        return terms
