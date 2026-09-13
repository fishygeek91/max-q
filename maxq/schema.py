"""Question and result schemas. See docs/methodology.md."""
from __future__ import annotations
from pydantic import BaseModel
from typing import Literal

Domain = Literal["propulsion", "orbital", "structures", "gnc", "telemetry"]
Difficulty = Literal["undergrad", "practitioner", "expert"]
ScoringMode = Literal["numeric_tolerance", "rubric", "exact", "ground_truth_series"]

class Question(BaseModel):
    id: str                      # e.g. "W1-PROP-007"
    domain: Domain
    difficulty: Difficulty
    prompt: str
    scoring: ScoringMode
    answer: str | float | None = None   # canonical answer (numeric or exact)
    unit: str | None = None             # pint-parseable unit for numeric answers
    rel_tol: float | None = None        # relative tolerance for numeric_tolerance
    rubric: list[str] | None = None     # rubric criteria for derivation questions
    provenance: str                     # how this question was authored (contamination note)

class ModelAnswer(BaseModel):
    question_id: str
    model: str
    attempt: int
    raw_response: str
    parsed_answer: str | float | None = None
    score: float | None = None          # 0.0–1.0
    scorer_notes: str | None = None
