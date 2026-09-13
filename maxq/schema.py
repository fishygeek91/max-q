"""Question, run-config, transcript, and scoring-result schemas.

See ``docs/methodology.md``. ``Question`` / ``ModelAnswer`` are unchanged from
Wave 1. Runner artifacts (``Transcript``, ``RunConfig``) are defined here so
scoring (issue #3) can read the same on-disk contract.
"""

from __future__ import annotations

from collections import Counter
from typing import Literal

from pydantic import BaseModel, Field, model_validator

Domain = Literal["propulsion", "orbital", "structures", "gnc", "telemetry"]
Difficulty = Literal["undergrad", "practitioner", "expert"]
ScoringMode = Literal["numeric_tolerance", "rubric", "exact", "ground_truth_series"]
ProviderName = Literal["xai", "anthropic", "openai", "google"]
ModelRole = Literal["flagship", "baseline", "small_floor"]
MessageRole = Literal["system", "user"]

TRANSCRIPT_KEY_ORDER: tuple[str, ...] = (
    "question_id",
    "provider",
    "requested_model",
    "response_model",
    "attempt",
    "wave",
    "prompt_template_id",
    "rendered_messages",
    "configured_temperature",
    "request_temperature",
    "max_output_tokens",
    "run_config_sha256",
    "text",
    "raw_request",
    "raw_response",
    "usage",
    "cost_usd",
    "sdk_version",
    "started_at",
    "finished_at",
    "dry_run",
    "truncated",
    "error",
)
"""JSON object key order locked for published transcripts."""

COST_LEDGER_KEY_ORDER: tuple[str, ...] = (
    "wave",
    "updated_at",
    "run_config_sha256",
    "dry_run",
    "enabled_model_ids",
    "per_model",
    "totals",
)


class Question(BaseModel):
    id: str  # e.g. "W1-PROP-007"
    domain: Domain
    difficulty: Difficulty
    prompt: str
    scoring: ScoringMode
    answer: str | float | None = None  # canonical answer (numeric or exact)
    unit: str | None = None  # pint-parseable unit for numeric answers
    rel_tol: float | None = None  # relative tolerance for numeric_tolerance
    rubric: list[str] | None = None  # rubric criteria for derivation questions
    provenance: str  # how this question was authored (contamination note)


class ModelAnswer(BaseModel):
    """Scoring-side record. Issue #3 reads ``Transcript.text``, not this, at run time."""

    question_id: str
    model: str
    attempt: int
    raw_response: str
    parsed_answer: str | float | None = None
    score: float | None = None  # 0.0–1.0
    scorer_notes: str | None = None


class TokenUsage(BaseModel):
    """Token counts used for cost accounting. Cached tokens are a subset of input."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0


class Pricing(BaseModel):
    """USD per million tokens. ``cached_input`` falls back to ``input`` when null."""

    input: float = Field(ge=0.0)
    output: float = Field(ge=0.0)
    cached_input: float | None = Field(default=None, ge=0.0)


class ModelSpec(BaseModel):
    """One row in the committed run config. A Grok 4.7 add is a new row, not code."""

    id: str
    provider: ProviderName
    enabled: bool = True
    role: ModelRole
    send_temperature: bool
    pricing_usd_per_mtok: Pricing
    max_output_tokens: int | None = Field(default=None, ge=1)
    """Per-model output-token budget override (reasoning models whose thinking
    tokens share the visible-output cap need more headroom). Null uses the
    run-level ``RunConfig.max_output_tokens``. Any override is a documented
    per-provider deviation from identical treatment — explain it in ``notes``."""
    notes: str = ""

    @model_validator(mode="after")
    def _id_nonempty(self) -> ModelSpec:
        if not self.id.strip():
            raise ValueError("model id must be non-empty")
        return self


class RunConfig(BaseModel):
    """Identical-treatment spec. Loaded from ``config/run.json``."""

    temperature: float = Field(ge=0.0)
    attempts: int = Field(ge=1)
    max_output_tokens: int = Field(ge=1)
    timeout_s: float = Field(gt=0.0)
    retry_max: int = Field(ge=1)
    retry_base_s: float = Field(ge=0.0)
    sleep_between_calls_s: float = Field(ge=0.0)
    prompt_template_id: str
    system_prompt: str
    user_suffix: str
    models: list[ModelSpec]

    @model_validator(mode="after")
    def _validate_treatment(self) -> RunConfig:
        if not self.prompt_template_id.strip():
            raise ValueError("prompt_template_id must be non-empty")
        if not self.system_prompt.strip():
            raise ValueError("system_prompt must be non-empty")
        if not self.user_suffix.strip():
            raise ValueError("user_suffix must be non-empty")
        if len(self.models) == 0:
            raise ValueError("models must be a non-empty list")
        ids = [spec.id for spec in self.models]
        duplicates = [model_id for model_id, count in Counter(ids).items() if count > 1]
        if duplicates:
            raise ValueError(f"duplicate model ids: {duplicates}")
        return self


class RenderedMessage(BaseModel):
    """Canonical message text shared across providers (wire shape may differ)."""

    role: MessageRole
    content: str


class Transcript(BaseModel):
    """On-disk run artifact: one (question, model, attempt) triple.

    Does not persist question answer keys, provenance, rubric, or API keys.
    ``text`` is the visible answer; thinking/reasoning stays in ``raw_response``.
    """

    question_id: str
    provider: ProviderName
    requested_model: str
    response_model: str | None
    attempt: int = Field(ge=1)
    wave: str
    prompt_template_id: str
    rendered_messages: list[RenderedMessage]
    configured_temperature: float
    request_temperature: float | None
    max_output_tokens: int
    run_config_sha256: str
    text: str
    raw_request: dict[str, object]
    raw_response: dict[str, object]
    usage: TokenUsage
    cost_usd: float
    sdk_version: str
    started_at: str
    finished_at: str
    dry_run: bool
    truncated: bool = False
    """True when the provider stopped at the output-token cap (finish/stop
    reason), so an absent FINAL line is a budget artifact, not a wrong answer.
    Scoring must report truncated attempts separately from incorrect ones."""
    error: str | None = None


class AdapterResult(BaseModel):
    """Normalized return value from a provider adapter ``complete`` call."""

    text: str
    usage: TokenUsage
    raw_request: dict[str, object]
    raw_response: dict[str, object]
    request_temperature: float | None
    response_model: str | None
    sdk_version: str
    truncated: bool = False


class ModelCostRow(BaseModel):
    """Per-model rollup inside ``cost.json``."""

    requested_model: str
    calls: int = 0
    skipped: int = 0
    errors: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cost_usd: float = 0.0


class CostTotals(BaseModel):
    """Grand totals for a wave cost ledger."""

    calls: int = 0
    skipped: int = 0
    errors: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cost_usd: float = 0.0


class CostLedger(BaseModel):
    """Rebuilt from transcripts on disk after every persist (never incremented)."""

    wave: str
    updated_at: str
    run_config_sha256: str
    dry_run: bool
    enabled_model_ids: list[str]
    per_model: list[ModelCostRow]
    totals: CostTotals
