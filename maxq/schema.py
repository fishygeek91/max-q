"""Question, run-config, transcript, and scoring-result schemas.

See ``docs/methodology.md``. Runner artifacts (``Transcript``, ``RunConfig``)
are defined here so scoring can read the same on-disk contract. ``ModelAnswer``
and the wave score report are the scored-JSON contract (issue #3).
"""

from __future__ import annotations

from collections import Counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Domain = Literal["propulsion", "orbital", "structures", "gnc", "telemetry"]
Difficulty = Literal["undergrad", "practitioner", "expert"]
ScoringMode = Literal["numeric_tolerance", "rubric", "exact", "ground_truth_series"]
AttemptStatus = Literal[
    "correct",
    "incorrect",
    "unparseable",
    "truncated",
    "error",
    "missing",
    "pending_rubric",
]
RubricQueueStatus = Literal["pending", "confirmed"]
OverrideAction = Literal["accept", "override"]
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

MODEL_ANSWER_KEY_ORDER: tuple[str, ...] = (
    "question_id",
    "model",
    "attempt",
    "raw_response",
    "parsed_answer",
    "score",
    "scorer_notes",
    "status",
    "scoring",
    "truncated",
    "pending_rubric",
    "criterion_scores",
)
"""JSON object key order locked for per-attempt rows inside scored.json."""

TIER_METRICS_KEY_ORDER: tuple[str, ...] = (
    "n_questions",
    "pass_at_1",
    "best_of_n",
    "truncated_attempts",
    "unparseable",
    "pending_rubric",
    "errors",
    "missing",
)

MODEL_SUMMARY_KEY_ORDER: tuple[str, ...] = (
    "model",
    "n_questions",
    "n_attempts",
    "pass_at_1",
    "best_of_n",
    "mean_score_at_1",
    "by_tier",
    "truncated_attempts",
    "unparseable",
    "pending_rubric",
    "errors",
    "missing",
)

WAVE_SCORE_KEY_ORDER: tuple[str, ...] = (
    "wave",
    "n_attempts",
    "models",
    "attempts",
)

RUBRIC_QUEUE_ITEM_KEY_ORDER: tuple[str, ...] = (
    "question_id",
    "model",
    "attempt",
    "rubric",
    "llm_scores",
    "llm_notes",
    "judge_model",
    "status",
    "confirmed_scores",
)

OVERRIDE_LOG_KEY_ORDER: tuple[str, ...] = (
    "action",
    "question_id",
    "model",
    "attempt",
    "criterion_index",
    "from",
    "to",
    "reviewer",
    "reason",
    "at",
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
    """Scoring-side record for one (question, model, attempt) triple.

    Built from ``Transcript.text`` plus the question key; never written back
    into the transcript file.
    """

    question_id: str
    model: str
    attempt: int = Field(ge=1)
    raw_response: str
    parsed_answer: str | float | None = None
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    scorer_notes: str | None = None
    status: AttemptStatus
    scoring: ScoringMode
    truncated: bool = False
    pending_rubric: bool = False
    criterion_scores: list[bool] | None = None


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


class RubricQueueItem(BaseModel):
    """One rubric attempt in ``rubric-queue.json`` awaiting or after human confirm."""

    question_id: str
    model: str
    attempt: int = Field(ge=1)
    rubric: list[str]
    llm_scores: list[bool] | None = None
    llm_notes: str | None = None
    judge_model: str | None = None
    """Model id that produced ``llm_scores`` ("stub" for the offline judge).
    Published with the queue so readers can audit whether a contestant judged
    itself. None only for rows that never received a first pass."""
    status: RubricQueueStatus = "pending"
    confirmed_scores: list[bool] | None = None


class RubricAcceptSpec(BaseModel):
    """Human accept-as-is of one pending queue row (``--accept-from``)."""

    question_id: str
    model: str
    attempt: int = Field(ge=1)


class RubricOverrideSpec(BaseModel):
    """Human edit applied via ``--apply-overrides`` (JSON array on disk)."""

    question_id: str
    model: str
    attempt: int = Field(ge=1)
    criterion_index: int = Field(ge=0)
    to: bool
    reviewer: str
    reason: str

    @model_validator(mode="after")
    def _nonempty_review(self) -> RubricOverrideSpec:
        if not self.reviewer.strip():
            raise ValueError("reviewer must be non-empty")
        if not self.reason.strip():
            raise ValueError("reason must be non-empty")
        return self


class OverrideLogLine(BaseModel):
    """One public audit record appended to ``overrides.jsonl``."""

    model_config = ConfigDict(populate_by_name=True)

    action: OverrideAction
    question_id: str
    model: str
    attempt: int = Field(ge=1)
    criterion_index: int | None = None
    from_value: bool | None = Field(default=None, alias="from")
    to: bool | None = None
    reviewer: str
    reason: str
    at: str


class TierMetrics(BaseModel):
    """pass@1 / best-of-n and unscored counts for one difficulty tier."""

    n_questions: int = Field(ge=0)
    pass_at_1: float = Field(ge=0.0, le=1.0)
    best_of_n: float = Field(ge=0.0, le=1.0)
    truncated_attempts: int = Field(default=0, ge=0)
    unparseable: int = Field(default=0, ge=0)
    pending_rubric: int = Field(default=0, ge=0)
    errors: int = Field(default=0, ge=0)
    missing: int = Field(default=0, ge=0)


class ModelScoreSummary(BaseModel):
    """Per-model rollup. Headline numbers are meaningless without ``by_tier``."""

    model: str
    n_questions: int = Field(ge=0)
    n_attempts: int = Field(ge=1)
    pass_at_1: float = Field(ge=0.0, le=1.0)
    best_of_n: float = Field(ge=0.0, le=1.0)
    mean_score_at_1: float | None = Field(default=None, ge=0.0, le=1.0)
    by_tier: dict[str, TierMetrics]
    truncated_attempts: int = Field(default=0, ge=0)
    unparseable: int = Field(default=0, ge=0)
    pending_rubric: int = Field(default=0, ge=0)
    errors: int = Field(default=0, ge=0)
    missing: int = Field(default=0, ge=0)


class WaveScoreReport(BaseModel):
    """On-disk ``scored.json``: per-attempt rows plus per-model, per-tier metrics."""

    wave: str
    n_attempts: int = Field(ge=1)
    models: list[ModelScoreSummary]
    attempts: list[ModelAnswer]
