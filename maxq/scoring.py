"""Scoring: mechanical wherever possible.

CLI::

    python -m maxq.scoring --questions PATH --wave SLUG

- numeric_tolerance: parse FINAL, convert units with pint, compare within rel_tol
- exact: stripped string match on the FINAL payload
- ground_truth_series: JSON array vs frozen precomputed truth (same unit + rel_tol)
- rubric: LLM-assisted first pass, human-confirmed via a file queue; log overrides
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pint
from pydantic import TypeAdapter, ValidationError

from maxq.providers import ProviderAdapter, adapter_for, call_with_retry, missing_key_errors
from maxq.runner import (
    atomic_write_json,
    default_config_path,
    load_run_config,
    ordered_dump,
    select_models,
    transcript_path,
    utc_now,
    wave_dirname,
)
from maxq.schema import (
    MODEL_ANSWER_KEY_ORDER,
    MODEL_SUMMARY_KEY_ORDER,
    OVERRIDE_LOG_KEY_ORDER,
    RUBRIC_QUEUE_ITEM_KEY_ORDER,
    TIER_METRICS_KEY_ORDER,
    WAVE_SCORE_KEY_ORDER,
    AttemptStatus,
    Difficulty,
    ModelAnswer,
    ModelScoreSummary,
    ModelSpec,
    OverrideLogLine,
    ProviderName,
    Question,
    RubricOverrideSpec,
    RubricQueueItem,
    RunConfig,
    ScoringMode,
    TierMetrics,
    Transcript,
    WaveScoreReport,
)
from maxq.wave import WaveError, load_questions

FINAL_LINE_RE = re.compile(r"^FINAL:\s*(.*)$", re.IGNORECASE)
JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)
JUDGE_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
TRAILING_PUNCT_RE = re.compile(r"[.,;:]+$")
TIER_ORDER: tuple[Difficulty, ...] = ("undergrad", "practitioner", "expert")
RUBRIC_SYSTEM = (
    "You are scoring an aerospace exam answer against a published rubric. "
    "Return ONLY JSON with this exact shape: "
    '{"criteria": [<one boolean per rubric bullet, in order>], "notes": "<short note>"}. '
    "Each criteria value is true only if that bullet is clearly satisfied. No extra keys."
)
_UREG = pint.UnitRegistry()
_OVERRIDE_ADAPTER = TypeAdapter(list[RubricOverrideSpec])
_PINT_ERRORS = (
    pint.PintError,
    ValueError,
    TypeError,
    ArithmeticError,
    AttributeError,
)


class RubricJudge(Protocol):
    """First-pass rubric scorer. Tests inject a fake; CLI may use a live adapter."""

    judge_id: str
    """Identity recorded in the published rubric queue (model id or "stub")."""

    def score(
        self,
        *,
        prompt: str,
        transcript_text: str,
        rubric: list[str],
    ) -> RubricJudgment:
        """Return one boolean per rubric bullet, plus a short note."""


@dataclass(frozen=True)
class RubricJudgment:
    """Per-bullet first-pass result from a ``RubricJudge``."""

    criterion_scores: list[bool]
    notes: str


class StubRubricJudge:
    """Offline judge used by ``--judge-stub``. Marks every bullet unmet."""

    judge_id = "stub"

    def score(
        self,
        *,
        prompt: str,
        transcript_text: str,
        rubric: list[str],
    ) -> RubricJudgment:
        """Return all-false scores so a human must confirm or override."""
        del prompt, transcript_text
        if len(rubric) == 0:
            raise WaveError("stub judge requires a non-empty rubric")
        return RubricJudgment(
            criterion_scores=[False] * len(rubric),
            notes="stub judge (no LLM)",
        )


class ProviderRubricJudge:
    """LLM first pass via an existing provider adapter. Not a contestant by default."""

    def __init__(
        self,
        *,
        adapter: ProviderAdapter,
        model_id: str,
        config: RunConfig,
        send_temperature: bool,
    ) -> None:
        if not model_id.strip():
            raise WaveError("judge model id must be non-empty")
        self._adapter = adapter
        self._model_id = model_id
        self._config = config
        self._send_temperature = send_temperature
        self.judge_id = model_id

    def score(
        self,
        *,
        prompt: str,
        transcript_text: str,
        rubric: list[str],
    ) -> RubricJudgment:
        """Call the judge model and parse a strict criteria JSON object."""
        if len(rubric) == 0:
            raise WaveError("provider judge requires a non-empty rubric")
        numbered = "\n".join(f"{index + 1}. {bullet}" for index, bullet in enumerate(rubric))
        user = (
            f"QUESTION:\n{prompt}\n\n"
            f"ANSWER:\n{transcript_text}\n\n"
            f"RUBRIC ({len(rubric)} bullets):\n{numbered}"
        )
        result = call_with_retry(
            self._adapter,
            retry_max=self._config.retry_max,
            retry_base_s=self._config.retry_base_s,
            sleep=True,
            model_id=self._model_id,
            system=RUBRIC_SYSTEM,
            user=user,
            temperature=self._config.temperature,
            send_temperature=self._send_temperature,
            max_output_tokens=self._config.max_output_tokens,
            timeout_s=self._config.timeout_s,
        )
        judgment = parse_judge_json(result.text, len(rubric))
        if judgment is None:
            raise WaveError("judge returned JSON that did not match the rubric")
        return judgment


def extract_final(text: str) -> str | None:
    """Return the payload of the last ``FINAL:`` line, or None if missing/empty."""
    last: str | None = None
    for raw_line in text.splitlines():
        match = FINAL_LINE_RE.match(raw_line.strip())
        if match is not None:
            last = match.group(1).strip()
            last = TRAILING_PUNCT_RE.sub("", last).strip()
    if last is None or last == "":
        return None
    return last


def parse_numeric(payload: str, unit: str) -> pint.Quantity | None:
    """Parse ``payload`` as a pint quantity; bare numbers use ``unit``.

    Returns None when the text is empty, not a quantity, or the fallback unit
    is not pint-parseable.
    """
    stripped = payload.strip()
    if stripped == "" or not unit.strip():
        return None
    try:
        parsed: object = _UREG(stripped)
    except _PINT_ERRORS:
        return None
    if not isinstance(parsed, _UREG.Quantity):
        try:
            magnitude = float(parsed)
        except (TypeError, ValueError):
            return None
        try:
            return _UREG.Quantity(magnitude, unit)
        except _PINT_ERRORS:
            return None
    quantity: pint.Quantity = parsed
    if _is_unitless(quantity):
        try:
            return _UREG.Quantity(float(quantity.magnitude), unit)
        except _PINT_ERRORS:
            return None
    return quantity


def within_rel_tol(candidate: pint.Quantity, truth: pint.Quantity, rel_tol: float) -> bool:
    """Return True iff ``candidate`` is within ``rel_tol`` of ``truth`` (inclusive).

    Quantities are converted to ``truth`` units first. A zero truth passes only
    when the converted candidate magnitude is also zero. Incompatible units
    return False; callers that need a scorer note should catch conversion first.
    """
    if rel_tol <= 0.0:
        return False
    try:
        converted = candidate.to(truth.units)
    except _PINT_ERRORS:
        return False
    candidate_value = float(converted.magnitude)
    truth_value = float(truth.magnitude)
    if truth_value == 0.0:
        return candidate_value == 0.0
    return abs(candidate_value - truth_value) <= rel_tol * abs(truth_value)


def parse_judge_json(text: str, n_criteria: int) -> RubricJudgment | None:
    """Parse ``{"criteria": [bool, ...], "notes": str}`` from judge visible text."""
    if n_criteria < 1:
        return None
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    payload: object
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        match = JUDGE_JSON_RE.search(stripped)
        if match is None:
            return None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(payload, dict):
        return None
    criteria_raw: object = payload.get("criteria")
    notes_raw: object = payload.get("notes", "")
    if not isinstance(criteria_raw, list) or len(criteria_raw) != n_criteria:
        return None
    scores: list[bool] = []
    for item in criteria_raw:
        if not isinstance(item, bool):
            return None
        scores.append(item)
    notes = notes_raw if isinstance(notes_raw, str) else ""
    return RubricJudgment(criterion_scores=scores, notes=notes)


def score_numeric(question: Question, transcript: Transcript) -> ModelAnswer:
    """Score a ``numeric_tolerance`` attempt from ``transcript.text``."""
    if question.scoring != "numeric_tolerance":
        raise WaveError(f"{question.id}: score_numeric called with scoring={question.scoring}")
    payload = extract_final(transcript.text)
    if payload is None:
        return _extraction_failure(
            question, transcript, "no parseable FINAL payload", parsed_answer=None
        )
    if (
        question.unit is None
        or question.rel_tol is None
        or isinstance(question.answer, bool)
        or not isinstance(question.answer, (int, float))
    ):
        return _extraction_failure(
            question,
            transcript,
            "question missing numeric answer, unit, or rel_tol",
            parsed_answer=None,
        )
    candidate = parse_numeric(payload, question.unit)
    if candidate is None:
        return _extraction_failure(
            question, transcript, "could not parse numeric quantity", parsed_answer=None
        )
    try:
        truth = _UREG.Quantity(float(question.answer), question.unit)
    except _PINT_ERRORS:
        return _extraction_failure(
            question, transcript, "question unit is not pint-parseable", parsed_answer=None
        )
    try:
        converted = candidate.to(truth.units)
    except _PINT_ERRORS:
        note = f"incompatible units: candidate {candidate.units} vs question {truth.units}"
        return _model_answer(
            question,
            transcript,
            parsed_answer=None,
            score=0.0,
            status="unparseable",
            notes=note,
        )
    parsed_value = float(converted.magnitude)
    passed = within_rel_tol(candidate, truth, question.rel_tol)
    if passed:
        note = _truncation_note(transcript, None)
        return _model_answer(
            question,
            transcript,
            parsed_answer=parsed_value,
            score=1.0,
            status="correct",
            notes=note,
        )
    truth_value = float(truth.magnitude)
    rel_err = None if truth_value == 0.0 else abs(parsed_value - truth_value) / abs(truth_value)
    detail = (
        f"relative error {rel_err} exceeds rel_tol {question.rel_tol}"
        if rel_err is not None
        else "candidate is non-zero against a zero truth"
    )
    return _model_answer(
        question,
        transcript,
        parsed_answer=parsed_value,
        score=0.0,
        status="incorrect",
        notes=_truncation_note(transcript, detail),
    )


def score_series(question: Question, transcript: Transcript) -> ModelAnswer:
    """Score a ``ground_truth_series`` attempt against the frozen JSON array."""
    if question.scoring != "ground_truth_series":
        raise WaveError(f"{question.id}: score_series called with scoring={question.scoring}")
    payload = extract_final(transcript.text)
    if payload is None:
        return _extraction_failure(
            question, transcript, "no parseable FINAL payload", parsed_answer=None
        )
    if question.unit is None or question.rel_tol is None:
        return _extraction_failure(
            question, transcript, "question missing unit or rel_tol", parsed_answer=None
        )
    truth_values = _parse_truth_series(question.answer)
    if truth_values is None:
        return _extraction_failure(
            question,
            transcript,
            "question series answer is not a numeric JSON array",
            parsed_answer=None,
        )
    candidate_values = _parse_series_payload(payload)
    if candidate_values is None:
        return _extraction_failure(
            question, transcript, "could not parse FINAL JSON array", parsed_answer=None
        )
    if len(candidate_values) != len(truth_values):
        note = (
            f"series length {len(candidate_values)} does not match truth length {len(truth_values)}"
        )
        return _model_answer(
            question,
            transcript,
            parsed_answer=json.dumps(candidate_values),
            score=0.0,
            status="incorrect",
            notes=_truncation_note(transcript, note),
        )
    converted: list[float] = []
    for index, raw_value in enumerate(candidate_values):
        candidate = parse_numeric(str(raw_value), question.unit)
        if candidate is None:
            return _extraction_failure(
                question,
                transcript,
                f"series element {index} is not a quantity",
                parsed_answer=json.dumps(candidate_values),
            )
        try:
            truth = _UREG.Quantity(float(truth_values[index]), question.unit)
            converted_q = candidate.to(truth.units)
        except _PINT_ERRORS:
            return _model_answer(
                question,
                transcript,
                parsed_answer=json.dumps(candidate_values),
                score=0.0,
                status="unparseable",
                notes=_truncation_note(transcript, f"incompatible units on series element {index}"),
            )
        converted.append(float(converted_q.magnitude))
        if not within_rel_tol(candidate, truth, question.rel_tol):
            note = f"series element {index} outside rel_tol {question.rel_tol}"
            return _model_answer(
                question,
                transcript,
                parsed_answer=json.dumps(converted),
                score=0.0,
                status="incorrect",
                notes=_truncation_note(transcript, note),
            )
    return _model_answer(
        question,
        transcript,
        parsed_answer=json.dumps(converted),
        score=1.0,
        status="correct",
        notes=_truncation_note(transcript, None),
    )


def score_exact(question: Question, transcript: Transcript) -> ModelAnswer:
    """Score an ``exact`` attempt with stripped, case-sensitive string equality."""
    if question.scoring != "exact":
        raise WaveError(f"{question.id}: score_exact called with scoring={question.scoring}")
    payload = extract_final(transcript.text)
    if payload is None:
        return _extraction_failure(
            question, transcript, "no parseable FINAL payload", parsed_answer=None
        )
    if not isinstance(question.answer, str) or not question.answer.strip():
        return _extraction_failure(
            question, transcript, "question missing exact-string answer", parsed_answer=payload
        )
    expected = question.answer.strip()
    passed = payload == expected
    return _model_answer(
        question,
        transcript,
        parsed_answer=payload,
        score=1.0 if passed else 0.0,
        status="correct" if passed else "incorrect",
        notes=_truncation_note(transcript, None),
    )


def score_rubric(
    question: Question,
    transcript: Transcript,
    *,
    criterion_scores: list[bool] | None,
    pending: bool,
    notes: str | None,
) -> ModelAnswer:
    """Build a rubric ``ModelAnswer`` from first-pass or confirmed criterion booleans.

    Pending rows keep ``score=None`` so they cannot inflate pass@1 / best-of-n.
    Confirmed rows use the mean of per-bullet 0/1; a pass still requires 1.0.
    """
    if question.scoring != "rubric":
        raise WaveError(f"{question.id}: score_rubric called with scoring={question.scoring}")
    if question.rubric is None or len(question.rubric) == 0:
        return _extraction_failure(
            question, transcript, "question missing rubric", parsed_answer=None
        )
    expected = len(question.rubric)
    if criterion_scores is not None and len(criterion_scores) != expected:
        return _extraction_failure(
            question,
            transcript,
            f"criterion_scores length {len(criterion_scores)} != rubric length {expected}",
            parsed_answer=None,
        )
    if pending:
        return _model_answer(
            question,
            transcript,
            parsed_answer=None,
            score=None,
            status="pending_rubric",
            notes=_truncation_note(transcript, notes),
            pending_rubric=True,
            criterion_scores=criterion_scores,
        )
    if criterion_scores is None:
        return _extraction_failure(
            question,
            transcript,
            "confirmed rubric row is missing criterion_scores",
            parsed_answer=None,
        )
    score = _mean_bools(criterion_scores)
    status: AttemptStatus = "correct" if score == 1.0 else "incorrect"
    return _model_answer(
        question,
        transcript,
        parsed_answer=None,
        score=score,
        status=status,
        notes=_truncation_note(transcript, notes),
        pending_rubric=False,
        criterion_scores=criterion_scores,
    )


def score_attempt(
    question: Question,
    transcript: Transcript | None,
    *,
    model_id: str,
    attempt: int,
    judge: RubricJudge | None = None,
    confirmed_scores: list[bool] | None = None,
) -> ModelAnswer:
    """Dispatch one triple. ``transcript`` None means a missing on-disk file."""
    if transcript is None:
        return _terminal_answer(
            question,
            model_id=model_id,
            attempt=attempt,
            raw_response="",
            status="missing",
            notes="transcript file missing or unreadable",
            truncated=False,
        )
    if transcript.error is not None:
        return _terminal_answer(
            question,
            model_id=model_id,
            attempt=attempt,
            raw_response=transcript.text,
            status="error",
            notes=transcript.error,
            truncated=transcript.truncated,
        )
    mode: ScoringMode = question.scoring
    if mode == "numeric_tolerance":
        return score_numeric(question, transcript)
    if mode == "ground_truth_series":
        return score_series(question, transcript)
    if mode == "exact":
        return score_exact(question, transcript)
    if mode == "rubric":
        return _score_rubric_with_judge(
            question,
            transcript,
            judge=judge,
            confirmed_scores=confirmed_scores,
        )
    raise WaveError(f"{question.id}: unsupported scoring mode {mode!r}")


def load_transcript_any(path: Path) -> Transcript | None:
    """Load a transcript including error rows. None if missing or invalid JSON."""
    if not path.is_file():
        return None
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
        return Transcript.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValidationError):
        return None


def score_wave(
    *,
    questions: list[Question],
    config: RunConfig,
    models: Sequence[ModelSpec],
    results_root: Path,
    wave_slug: str,
    judge: RubricJudge | None = None,
    accept_llm: bool = False,
    overrides: Sequence[RubricOverrideSpec] | None = None,
    reviewer: str = "operator",
) -> WaveScoreReport:
    """Score every (model, question, attempt) triple and persist scored artifacts."""
    if len(questions) == 0:
        raise WaveError("wave is empty")
    if len(models) == 0:
        raise WaveError("no models selected")
    model_ids = [spec.id for spec in models]
    wave = wave_dirname(wave_slug)
    wave_dir = results_root / wave
    queue_path = wave_dir / "rubric-queue.json"
    overrides_path = wave_dir / "overrides.jsonl"
    scored_path = wave_dir / "scored.json"

    queue = _load_rubric_queue(queue_path)
    log_lines: list[OverrideLogLine] = []
    _ensure_rubric_first_pass(
        questions=questions,
        model_ids=model_ids,
        config=config,
        results_root=results_root,
        wave=wave,
        queue=queue,
        judge=judge,
    )
    if overrides is not None:
        _apply_overrides(queue, overrides, log_lines)
    if accept_llm:
        _accept_llm(queue, reviewer=reviewer, log_lines=log_lines)

    attempts = _score_all_attempts(
        questions=questions,
        model_ids=model_ids,
        config=config,
        results_root=results_root,
        wave=wave,
        queue=queue,
    )
    summaries = [
        summarize_model(model_id, questions, attempts, config.attempts) for model_id in model_ids
    ]
    report = WaveScoreReport(
        wave=wave,
        n_attempts=config.attempts,
        models=summaries,
        attempts=attempts,
    )
    dump_score_report(report, scored_path)
    dump_rubric_queue(list(queue.values()), queue_path)
    _append_overrides(overrides_path, log_lines)
    return report


def summarize_model(
    model_id: str,
    questions: list[Question],
    attempts: Sequence[ModelAnswer],
    n_attempts: int,
) -> ModelScoreSummary:
    """Build per-model pass@1 / best-of-n plus a breakdown for every tier."""
    rows = [row for row in attempts if row.model == model_id]
    overall = _metrics_for(questions, rows)
    by_tier: dict[str, TierMetrics] = {}
    for tier in TIER_ORDER:
        tier_questions = [question for question in questions if question.difficulty == tier]
        by_tier[tier] = _metrics_for(tier_questions, rows)
    mean_at_1 = _mean_score_at_1(rows)
    return ModelScoreSummary(
        model=model_id,
        n_questions=overall.n_questions,
        n_attempts=n_attempts,
        pass_at_1=overall.pass_at_1,
        best_of_n=overall.best_of_n,
        mean_score_at_1=mean_at_1,
        by_tier=by_tier,
        truncated_attempts=overall.truncated_attempts,
        unparseable=overall.unparseable,
        pending_rubric=overall.pending_rubric,
        errors=overall.errors,
        missing=overall.missing,
    )


def format_score_table(report: WaveScoreReport) -> str:
    """Return a model×tier table. Never a single headline number."""
    header = "model\ttier\tn\tpass@1\tbest-of-n\ttruncated\tunparseable\tpending_rubric"
    lines = [header]
    for summary in report.models:
        for tier in TIER_ORDER:
            metrics = summary.by_tier.get(tier)
            if metrics is None:
                metrics = _empty_tier_metrics()
            lines.append(
                "\t".join(
                    [
                        summary.model,
                        tier,
                        str(metrics.n_questions),
                        f"{metrics.pass_at_1:.3f}",
                        f"{metrics.best_of_n:.3f}",
                        str(metrics.truncated_attempts),
                        str(metrics.unparseable),
                        str(metrics.pending_rubric),
                    ]
                )
            )
    return "\n".join(lines) + "\n"


def dump_score_report(report: WaveScoreReport, path: Path) -> None:
    """Persist ``scored.json`` with locked key order."""
    models_payload: list[dict[str, object]] = []
    for summary in report.models:
        dumped = summary.model_dump(mode="json")
        by_tier_obj: dict[str, object] = {}
        for tier in TIER_ORDER:
            metrics = summary.by_tier.get(tier)
            if metrics is None:
                metrics = _empty_tier_metrics()
            by_tier_obj[tier] = ordered_dump(
                metrics.model_dump(mode="json"), TIER_METRICS_KEY_ORDER
            )
        dumped["by_tier"] = by_tier_obj
        models_payload.append(ordered_dump(dumped, MODEL_SUMMARY_KEY_ORDER))
    attempts_payload = [
        ordered_dump(row.model_dump(mode="json"), MODEL_ANSWER_KEY_ORDER) for row in report.attempts
    ]
    payload: dict[str, object] = {
        "wave": report.wave,
        "n_attempts": report.n_attempts,
        "models": models_payload,
        "attempts": attempts_payload,
    }
    atomic_write_json(path, ordered_dump(payload, WAVE_SCORE_KEY_ORDER))


def dump_rubric_queue(items: Sequence[RubricQueueItem], path: Path) -> None:
    """Write the human-confirm queue, sorted by question / model / attempt."""
    ordered_items = sorted(items, key=lambda item: (item.question_id, item.model, item.attempt))
    payload: list[dict[str, object]] = [
        ordered_dump(item.model_dump(mode="json"), RUBRIC_QUEUE_ITEM_KEY_ORDER)
        for item in ordered_items
    ]
    atomic_write_json(path, payload)


def main(argv: list[str] | None = None, *, judge: RubricJudge | None = None) -> int:
    """CLI entry: score a wave directory of transcripts against a question file."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    config_path = args.config if args.config is not None else default_config_path()
    try:
        config = load_run_config(config_path)
        models = select_models(config, args.models)
        questions = load_questions(args.questions)
        if len(questions) == 0:
            raise WaveError("wave is empty")
        if len(models) == 0:
            raise WaveError("no enabled models selected")
        resolved_judge = _resolve_cli_judge(
            config=config,
            questions=questions,
            judge=judge,
            judge_model=args.judge_model,
            judge_provider=args.judge_provider,
            judge_stub=args.judge_stub,
        )
        override_specs = (
            _load_override_specs(args.apply_overrides) if args.apply_overrides is not None else None
        )
        report = score_wave(
            questions=questions,
            config=config,
            models=models,
            results_root=args.results_root,
            wave_slug=args.wave,
            judge=resolved_judge,
            accept_llm=args.accept_llm,
            overrides=override_specs,
            reviewer=args.reviewer,
        )
    except (OSError, WaveError, ValidationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(format_score_table(report), end="")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score transcripts: mechanical checkers plus a rubric confirm queue."
    )
    parser.add_argument(
        "--questions", type=Path, required=True, help="JSON array of Question objects"
    )
    parser.add_argument("--wave", required=True, help="Wave slug (1 → wave-1, smoke → wave-smoke)")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Run config JSON (default: repo config/run.json)",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Directory that contains wave-N/ folders",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        default=None,
        help="Enabled model id to include (repeatable)",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Model id for rubric first pass (prefer a model that is not a contestant)",
    )
    parser.add_argument(
        "--judge-provider",
        default=None,
        choices=["xai", "anthropic", "openai", "google"],
        help="Provider for --judge-model when it is not a run-config row",
    )
    parser.add_argument(
        "--judge-stub",
        action="store_true",
        help="Use an all-false stub judge (no API). For tests and dry scoring only.",
    )
    parser.add_argument(
        "--accept-llm",
        action="store_true",
        help="Confirm all pending rubric rows that have LLM (or stub) scores",
    )
    parser.add_argument(
        "--apply-overrides",
        type=Path,
        default=None,
        help="JSON array of RubricOverrideSpec objects to apply and log",
    )
    parser.add_argument(
        "--reviewer",
        default="operator",
        help="Name recorded on accept/override log lines",
    )
    return parser


def _resolve_cli_judge(
    *,
    config: RunConfig,
    questions: list[Question],
    judge: RubricJudge | None,
    judge_model: str | None,
    judge_provider: str | None,
    judge_stub: bool,
) -> RubricJudge | None:
    """Pick the judge injected by tests, a stub, a live adapter, or None."""
    if judge is not None:
        return judge
    has_rubric = any(question.scoring == "rubric" for question in questions)
    if not has_rubric:
        return None
    if judge_stub:
        return StubRubricJudge()
    if judge_model is None:
        return None
    provider_name, send_temperature = _judge_provider(config, judge_model, judge_provider)
    missing = missing_key_errors(provider_name, judge_model)
    if missing is not None:
        raise WaveError(missing)
    contestants = {spec.id for spec in config.models if spec.enabled}
    if judge_model in contestants:
        print(
            f"warning: judge model {judge_model!r} is also a contestant; "
            "its first-pass scores on its own answers are self-judged — "
            "confirm those rows by hand, and say so in the writeup",
            file=sys.stderr,
        )
    return ProviderRubricJudge(
        adapter=adapter_for(provider_name, dry_run=False),
        model_id=judge_model,
        config=config,
        send_temperature=send_temperature,
    )


def _judge_id(judge: RubricJudge) -> str:
    """Best-effort judge identity for the audit trail."""
    value = getattr(judge, "judge_id", None)
    if isinstance(value, str) and value.strip():
        return value
    return type(judge).__name__


def _judge_provider(
    config: RunConfig,
    judge_model: str,
    judge_provider: str | None,
) -> tuple[ProviderName, bool]:
    """Resolve provider + send_temperature for a judge model id."""
    for spec in config.models:
        if spec.id == judge_model:
            if judge_provider is not None and judge_provider != spec.provider:
                raise WaveError(
                    f"judge model {judge_model!r} is provider {spec.provider}, not {judge_provider}"
                )
            return spec.provider, spec.send_temperature
    if judge_provider is None:
        raise WaveError(
            f"judge model {judge_model!r} is not in the run config; pass --judge-provider"
        )
    return _as_provider(judge_provider), True


def _as_provider(value: str) -> ProviderName:
    if value in {"xai", "anthropic", "openai", "google"}:
        return value
    raise WaveError(f"unsupported judge provider {value!r}")


def _ensure_rubric_first_pass(
    *,
    questions: list[Question],
    model_ids: list[str],
    config: RunConfig,
    results_root: Path,
    wave: str,
    queue: dict[tuple[str, str, int], RubricQueueItem],
    judge: RubricJudge | None,
) -> None:
    """Fill pending queue rows with LLM/stub scores when a judge is available."""
    for model_id in model_ids:
        for question in questions:
            if question.scoring != "rubric":
                continue
            if question.rubric is None or len(question.rubric) == 0:
                raise WaveError(f"{question.id}: rubric scoring requires a non-empty rubric")
            for attempt in range(1, config.attempts + 1):
                key = (question.id, model_id, attempt)
                item = queue.get(key)
                if item is not None and item.status == "confirmed":
                    continue
                path = transcript_path(results_root, wave, model_id, question.id, attempt)
                transcript = load_transcript_any(path)
                if transcript is None or transcript.error is not None:
                    continue
                if transcript.truncated and not transcript.text.strip():
                    continue
                llm_scores = item.llm_scores if item is not None else None
                llm_notes = item.llm_notes if item is not None else None
                judge_model = item.judge_model if item is not None else None
                if llm_scores is None and judge is not None:
                    llm_scores, llm_notes = _run_judge(judge, question, transcript)
                    judge_model = _judge_id(judge)
                if item is None:
                    queue[key] = RubricQueueItem(
                        question_id=question.id,
                        model=model_id,
                        attempt=attempt,
                        rubric=list(question.rubric),
                        llm_scores=llm_scores,
                        llm_notes=llm_notes,
                        judge_model=judge_model,
                        status="pending",
                        confirmed_scores=None,
                    )
                else:
                    item.llm_scores = llm_scores
                    item.llm_notes = llm_notes
                    item.judge_model = judge_model
                    item.rubric = list(question.rubric)


def _run_judge(
    judge: RubricJudge,
    question: Question,
    transcript: Transcript,
) -> tuple[list[bool] | None, str]:
    """Call ``judge`` and return (scores, notes); scores None on failure."""
    rubric = question.rubric
    if rubric is None:
        return None, "question missing rubric"
    try:
        judgment = judge.score(
            prompt=question.prompt,
            transcript_text=transcript.text,
            rubric=list(rubric),
        )
    except Exception as exc:  # noqa: BLE001 - judge failures become queue notes
        return None, f"judge error: {type(exc).__name__}: {exc}"
    if len(judgment.criterion_scores) != len(rubric):
        return None, "judge returned the wrong number of criteria"
    return list(judgment.criterion_scores), judgment.notes


def _apply_overrides(
    queue: dict[tuple[str, str, int], RubricQueueItem],
    overrides: Sequence[RubricOverrideSpec],
    log_lines: list[OverrideLogLine],
) -> None:
    """Apply per-criterion human edits and mark those rows confirmed."""
    timestamp = utc_now()
    for spec in overrides:
        key = (spec.question_id, spec.model, spec.attempt)
        item = queue.get(key)
        if item is None:
            raise WaveError(
                f"override for unknown rubric attempt "
                f"{spec.question_id} {spec.model} a{spec.attempt}"
            )
        n_criteria = len(item.rubric)
        if spec.criterion_index >= n_criteria:
            raise WaveError(
                f"{spec.question_id}: criterion_index {spec.criterion_index} "
                f"out of range for rubric of length {n_criteria}"
            )
        base = _scores_for_override(item, n_criteria)
        previous = base[spec.criterion_index]
        base[spec.criterion_index] = spec.to
        item.confirmed_scores = base
        item.status = "confirmed"
        log_lines.append(
            OverrideLogLine(
                action="override",
                question_id=spec.question_id,
                model=spec.model,
                attempt=spec.attempt,
                criterion_index=spec.criterion_index,
                from_value=previous,
                to=spec.to,
                reviewer=spec.reviewer,
                reason=spec.reason,
                at=timestamp,
            )
        )


def _accept_llm(
    queue: dict[tuple[str, str, int], RubricQueueItem],
    *,
    reviewer: str,
    log_lines: list[OverrideLogLine],
) -> None:
    """Confirm pending rows that already have first-pass scores."""
    timestamp = utc_now()
    reviewer_name = reviewer.strip() if reviewer.strip() else "operator"
    for item in queue.values():
        if item.status != "pending":
            continue
        if item.llm_scores is None:
            continue
        item.confirmed_scores = list(item.llm_scores)
        item.status = "confirmed"
        log_lines.append(
            OverrideLogLine(
                action="accept",
                question_id=item.question_id,
                model=item.model,
                attempt=item.attempt,
                criterion_index=None,
                from_value=None,
                to=None,
                reviewer=reviewer_name,
                reason="accept-llm",
                at=timestamp,
            )
        )


def _score_all_attempts(
    *,
    questions: list[Question],
    model_ids: list[str],
    config: RunConfig,
    results_root: Path,
    wave: str,
    queue: Mapping[tuple[str, str, int], RubricQueueItem],
) -> list[ModelAnswer]:
    """Produce one ``ModelAnswer`` per configured triple."""
    rows: list[ModelAnswer] = []
    for model_id in model_ids:
        for question in questions:
            for attempt in range(1, config.attempts + 1):
                path = transcript_path(results_root, wave, model_id, question.id, attempt)
                transcript = load_transcript_any(path)
                if question.scoring == "rubric":
                    rows.append(
                        _rubric_from_queue(
                            question,
                            transcript,
                            model_id=model_id,
                            attempt=attempt,
                            item=queue.get((question.id, model_id, attempt)),
                        )
                    )
                    continue
                rows.append(
                    score_attempt(
                        question,
                        transcript,
                        model_id=model_id,
                        attempt=attempt,
                    )
                )
    return rows


def _rubric_from_queue(
    question: Question,
    transcript: Transcript | None,
    *,
    model_id: str,
    attempt: int,
    item: RubricQueueItem | None,
) -> ModelAnswer:
    """Map queue + transcript state onto a rubric ``ModelAnswer``."""
    if transcript is None:
        return _terminal_answer(
            question,
            model_id=model_id,
            attempt=attempt,
            raw_response="",
            status="missing",
            notes="transcript file missing or unreadable",
            truncated=False,
        )
    if transcript.error is not None:
        return _terminal_answer(
            question,
            model_id=model_id,
            attempt=attempt,
            raw_response=transcript.text,
            status="error",
            notes=transcript.error,
            truncated=transcript.truncated,
        )
    if transcript.truncated and not transcript.text.strip():
        return _terminal_answer(
            question,
            model_id=model_id,
            attempt=attempt,
            raw_response=transcript.text,
            status="truncated",
            notes="truncated before any visible text",
            truncated=True,
        )
    if item is not None and item.status == "confirmed" and item.confirmed_scores is not None:
        notes = item.llm_notes
        return score_rubric(
            question,
            transcript,
            criterion_scores=item.confirmed_scores,
            pending=False,
            notes=notes,
        )
    notes = None if item is None else item.llm_notes
    scores = None if item is None else item.llm_scores
    return score_rubric(
        question,
        transcript,
        criterion_scores=scores,
        pending=True,
        notes=notes,
    )


def _score_rubric_with_judge(
    question: Question,
    transcript: Transcript,
    *,
    judge: RubricJudge | None,
    confirmed_scores: list[bool] | None,
) -> ModelAnswer:
    """Used by ``score_attempt`` for isolated rubric tests."""
    if confirmed_scores is not None:
        return score_rubric(
            question,
            transcript,
            criterion_scores=confirmed_scores,
            pending=False,
            notes=None,
        )
    if transcript.truncated and not transcript.text.strip():
        return _terminal_answer(
            question,
            model_id=transcript.requested_model,
            attempt=transcript.attempt,
            raw_response=transcript.text,
            status="truncated",
            notes="truncated before any visible text",
            truncated=True,
        )
    notes: str | None = None
    scores: list[bool] | None = None
    if judge is not None:
        scores, notes = _run_judge(judge, question, transcript)
    return score_rubric(
        question,
        transcript,
        criterion_scores=scores,
        pending=True,
        notes=notes,
    )


def _metrics_for(
    questions: list[Question],
    rows: Sequence[ModelAnswer],
) -> TierMetrics:
    """Compute pass rates and unscored counts for a question subset."""
    n_questions = len(questions)
    ids = {question.id for question in questions}
    subset = [row for row in rows if row.question_id in ids]
    by_question: dict[str, list[ModelAnswer]] = {question.id: [] for question in questions}
    for row in subset:
        by_question.setdefault(row.question_id, []).append(row)
    pass_at_1 = 0
    best_of_n = 0
    for question in questions:
        group = by_question.get(question.id, [])
        attempt_one = next((row for row in group if row.attempt == 1), None)
        if attempt_one is not None and _is_pass(attempt_one):
            pass_at_1 += 1
        if any(_is_pass(row) for row in group):
            best_of_n += 1
    denom = float(n_questions) if n_questions > 0 else 1.0
    return TierMetrics(
        n_questions=n_questions,
        pass_at_1=(pass_at_1 / denom) if n_questions > 0 else 0.0,
        best_of_n=(best_of_n / denom) if n_questions > 0 else 0.0,
        truncated_attempts=sum(1 for row in subset if row.status == "truncated"),
        unparseable=sum(1 for row in subset if row.status == "unparseable"),
        pending_rubric=sum(1 for row in subset if row.status == "pending_rubric"),
        errors=sum(1 for row in subset if row.status == "error"),
        missing=sum(1 for row in subset if row.status == "missing"),
    )


def _mean_score_at_1(rows: Sequence[ModelAnswer]) -> float | None:
    """Mean of attempt-1 scores that are not None; None if none are scored."""
    values = [row.score for row in rows if row.attempt == 1 and row.score is not None]
    if len(values) == 0:
        return None
    return sum(values) / len(values)


def _is_pass(row: ModelAnswer) -> bool:
    return row.score == 1.0


def _empty_tier_metrics() -> TierMetrics:
    return TierMetrics(
        n_questions=0,
        pass_at_1=0.0,
        best_of_n=0.0,
    )


def _is_unitless(quantity: pint.Quantity) -> bool:
    """True for a bare number, not for named dimensionless units like percent."""
    if hasattr(quantity, "unitless"):
        return bool(quantity.unitless)
    return str(quantity.units) == "dimensionless"


def _load_rubric_queue(path: Path) -> dict[tuple[str, str, int], RubricQueueItem]:
    if not path.is_file():
        return {}
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise WaveError(f"{path}: invalid JSON ({exc})") from exc
    if not isinstance(payload, list):
        raise WaveError(f"{path}: rubric queue must be a JSON array")
    items: dict[tuple[str, str, int], RubricQueueItem] = {}
    for entry in payload:
        try:
            item = RubricQueueItem.model_validate(entry)
        except ValidationError as exc:
            raise WaveError(f"{path}: rubric queue validation failed:\n{exc}") from exc
        items[(item.question_id, item.model, item.attempt)] = item
    return items


def _load_override_specs(path: Path) -> list[RubricOverrideSpec]:
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise WaveError(f"{path}: invalid JSON ({exc})") from exc
    try:
        return _OVERRIDE_ADAPTER.validate_python(payload)
    except ValidationError as exc:
        raise WaveError(f"{path}: override spec validation failed:\n{exc}") from exc


def _append_overrides(path: Path, lines: Sequence[OverrideLogLine]) -> None:
    if len(lines) == 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for line in lines:
            dumped = ordered_dump(
                line.model_dump(mode="json", by_alias=True), OVERRIDE_LOG_KEY_ORDER
            )
            handle.write(json.dumps(dumped, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _scores_for_override(item: RubricQueueItem, n_criteria: int) -> list[bool]:
    if item.confirmed_scores is not None and len(item.confirmed_scores) == n_criteria:
        return list(item.confirmed_scores)
    if item.llm_scores is not None and len(item.llm_scores) == n_criteria:
        return list(item.llm_scores)
    return [False] * n_criteria


def _parse_truth_series(answer: str | float | None) -> list[float] | None:
    if not isinstance(answer, str):
        return None
    try:
        parsed: object = json.loads(answer)
    except json.JSONDecodeError:
        return None
    return _numeric_list(parsed)


def _parse_series_payload(payload: str) -> list[float] | None:
    stripped = payload.strip()
    try:
        parsed: object = json.loads(stripped)
    except json.JSONDecodeError:
        match = JSON_ARRAY_RE.search(stripped)
        if match is None:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return _numeric_list(parsed)


def _numeric_list(parsed: object) -> list[float] | None:
    if not isinstance(parsed, list) or len(parsed) == 0:
        return None
    values: list[float] = []
    for item in parsed:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return None
        values.append(float(item))
    return values


def _mean_bools(scores: Sequence[bool]) -> float:
    if len(scores) == 0:
        return 0.0
    return sum(1.0 if item else 0.0 for item in scores) / len(scores)


def _extraction_failure(
    question: Question,
    transcript: Transcript,
    note: str,
    *,
    parsed_answer: str | float | None,
) -> ModelAnswer:
    """Unparseable → score 0, unless the attempt was truncated at the token cap."""
    if transcript.truncated:
        return _model_answer(
            question,
            transcript,
            parsed_answer=parsed_answer,
            score=None,
            status="truncated",
            notes=_truncation_note(transcript, note),
        )
    return _model_answer(
        question,
        transcript,
        parsed_answer=parsed_answer,
        score=0.0,
        status="unparseable",
        notes=note,
    )


def _truncation_note(transcript: Transcript, note: str | None) -> str | None:
    if not transcript.truncated:
        return note
    prefix = "truncated attempt"
    if note is None or note == "":
        return prefix
    return f"{prefix}; {note}"


def _model_answer(
    question: Question,
    transcript: Transcript,
    *,
    parsed_answer: str | float | None,
    score: float | None,
    status: AttemptStatus,
    notes: str | None,
    pending_rubric: bool = False,
    criterion_scores: list[bool] | None = None,
) -> ModelAnswer:
    return ModelAnswer(
        question_id=question.id,
        model=transcript.requested_model,
        attempt=transcript.attempt,
        raw_response=transcript.text,
        parsed_answer=parsed_answer,
        score=score,
        scorer_notes=notes,
        status=status,
        scoring=question.scoring,
        truncated=transcript.truncated,
        pending_rubric=pending_rubric,
        criterion_scores=criterion_scores,
    )


def _terminal_answer(
    question: Question,
    *,
    model_id: str,
    attempt: int,
    raw_response: str,
    status: AttemptStatus,
    notes: str,
    truncated: bool,
) -> ModelAnswer:
    return ModelAnswer(
        question_id=question.id,
        model=model_id,
        attempt=attempt,
        raw_response=raw_response,
        parsed_answer=None,
        score=None,
        scorer_notes=notes,
        status=status,
        scoring=question.scoring,
        truncated=truncated,
        pending_rubric=False,
        criterion_scores=None,
    )


if __name__ == "__main__":
    sys.exit(main())
