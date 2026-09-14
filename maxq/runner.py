"""Run harness: identical treatment, resumable transcripts, cost rebuilt from disk.

CLI::

    python -m maxq.runner --questions PATH --wave SLUG [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from maxq.providers import (
    adapter_for,
    call_with_retry,
    missing_key_errors,
)
from maxq.schema import (
    COST_LEDGER_KEY_ORDER,
    TRANSCRIPT_KEY_ORDER,
    AdapterResult,
    CostLedger,
    CostTotals,
    ModelCostRow,
    ModelSpec,
    Pricing,
    Question,
    RenderedMessage,
    RunConfig,
    TokenUsage,
    Transcript,
)
from maxq.wave import WaveError, load_questions

WAVE_DIR_RE = re.compile(r"^wave-[A-Za-z0-9._-]+$")
SLUG_RE = re.compile(r"^[A-Za-z0-9._-]+$")
VERIFY_SYSTEM = "Reply with exactly one character."
VERIFY_USER = "Reply with the single character: y"
VERIFY_MAX_TOKENS = 1024
"""Ping budget. Reasoning models spend tokens on thinking before any visible
text, so a 1-token cap false-fails exactly the models under test. Verify
passes on any successful response (even with empty visible text)."""


def default_config_path() -> Path:
    """Return ``<repo>/config/run.json`` next to the ``maxq`` package."""
    return Path(__file__).resolve().parents[1] / "config" / "run.json"


def load_run_config(path: Path) -> RunConfig:
    """Load and validate the committed run config from ``path``."""
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise WaveError(f"{path}: invalid JSON ({exc})") from exc
    except OSError as exc:
        raise WaveError(f"{path}: {exc}") from exc
    try:
        return RunConfig.model_validate(payload)
    except ValidationError as exc:
        raise WaveError(f"{path}: run config validation failed:\n{exc}") from exc


def run_config_sha256(path: Path) -> str:
    """SHA-256 of the config file as stored on disk (do not re-serialize)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wave_dirname(slug: str) -> str:
    """Map ``--wave 1`` to ``wave-1`` and ``--wave smoke`` to ``wave-smoke``."""
    if not SLUG_RE.fullmatch(slug):
        raise WaveError(f"wave slug {slug!r} must match {SLUG_RE.pattern}")
    name = slug if slug.startswith("wave-") else f"wave-{slug}"
    if WAVE_DIR_RE.fullmatch(name) is None:
        raise WaveError(f"wave directory {name!r} is not filesystem-safe")
    return name


def model_slug(model_id: str) -> str:
    """Filesystem-safe directory name; ``/`` becomes ``--``."""
    replaced = model_id.replace("/", "--")
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "-", replaced)
    if not cleaned:
        raise WaveError(f"model id {model_id!r} is not filesystem-safe")
    return cleaned


def transcript_path(
    results_root: Path,
    wave: str,
    model_id: str,
    question_id: str,
    attempt: int,
) -> Path:
    """``{results_root}/wave-N/<model>/{question_id}-a{attempt}.json``."""
    return results_root / wave / model_slug(model_id) / f"{question_id}-a{attempt}.json"


def effective_max_output_tokens(spec: ModelSpec, config: RunConfig) -> int:
    """Per-model output-token budget: spec override, else the run-level value."""
    if spec.max_output_tokens is not None:
        return spec.max_output_tokens
    return config.max_output_tokens


def render_prompt(question: Question, config: RunConfig) -> list[RenderedMessage]:
    """Build the canonical system/user pair. Does not inject answer keys."""
    user = question.prompt + "\n" + config.user_suffix
    return [
        RenderedMessage(role="system", content=config.system_prompt),
        RenderedMessage(role="user", content=user),
    ]


def estimate_cost_usd(usage: TokenUsage, pricing: Pricing) -> float:
    """USD from token counts and per-million rates. Cached falls back to input rate."""
    cached_rate = pricing.cached_input if pricing.cached_input is not None else pricing.input
    cached = min(usage.cached_input_tokens, usage.input_tokens)
    uncached = usage.input_tokens - cached
    raw = (
        uncached * pricing.input + cached * cached_rate + usage.output_tokens * pricing.output
    ) / 1_000_000.0
    return round(raw, 8)


def select_models(config: RunConfig, only: list[str] | None) -> list[ModelSpec]:
    """Enabled models, optionally filtered by ``--model`` ids (order preserved)."""
    enabled = [spec for spec in config.models if spec.enabled]
    if only is None or len(only) == 0:
        return enabled
    by_id = {spec.id: spec for spec in enabled}
    missing = [model_id for model_id in only if model_id not in by_id]
    if missing:
        raise WaveError(f"unknown or disabled model ids: {missing}")
    return [by_id[model_id] for model_id in only]


def ordered_dump(payload: dict[str, object], key_order: tuple[str, ...]) -> dict[str, object]:
    """Return ``payload`` with ``key_order`` first, then any leftover keys."""
    ordered: dict[str, object] = {}
    for key in key_order:
        if key in payload:
            ordered[key] = payload[key]
    for key, value in payload.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def atomic_write_json(path: Path, payload: object) -> None:
    """Write JSON via ``.tmp`` + fsync + replace so resume never sees a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def dump_transcript(transcript: Transcript, path: Path) -> None:
    """Persist a transcript with locked key order."""
    dumped = transcript.model_dump(mode="json")
    atomic_write_json(path, ordered_dump(dumped, TRANSCRIPT_KEY_ORDER))


def load_complete_transcript(path: Path) -> Transcript | None:
    """Return a valid error-free transcript, else None (caller should retry)."""
    if not path.is_file():
        return None
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
        transcript = Transcript.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValidationError):
        return None
    if transcript.error is not None:
        return None
    return transcript


def utc_now() -> str:
    """UTC timestamp with a trailing Z."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def rebuild_cost_ledger(
    *,
    wave_dir: Path,
    wave: str,
    config_sha: str,
    dry_run: bool,
    enabled_model_ids: list[str],
    skipped_by_model: dict[str, int],
) -> CostLedger:
    """Rebuild ``cost.json`` from transcripts on disk (never increment a running total)."""
    rows: dict[str, ModelCostRow] = {}

    def row_for(model_id: str) -> ModelCostRow:
        if model_id not in rows:
            rows[model_id] = ModelCostRow(requested_model=model_id)
        return rows[model_id]

    for model_id in enabled_model_ids:
        row_for(model_id)
    for model_id, skipped in skipped_by_model.items():
        row_for(model_id).skipped += skipped

    if wave_dir.is_dir():
        for path in sorted(wave_dir.glob("*/*-a*.json")):
            if path.name.endswith(".tmp"):
                continue
            try:
                payload: object = json.loads(path.read_text(encoding="utf-8"))
                transcript = Transcript.model_validate(payload)
            except (OSError, json.JSONDecodeError, ValidationError):
                continue
            row = row_for(transcript.requested_model)
            if transcript.error is not None:
                row.errors += 1
                continue
            row.calls += 1
            row.input_tokens += transcript.usage.input_tokens
            row.output_tokens += transcript.usage.output_tokens
            row.cached_input_tokens += transcript.usage.cached_input_tokens
            row.cost_usd = round(row.cost_usd + transcript.cost_usd, 8)

    per_model = [rows[key] for key in sorted(rows)]
    totals = CostTotals()
    for row in per_model:
        totals.calls += row.calls
        totals.skipped += row.skipped
        totals.errors += row.errors
        totals.input_tokens += row.input_tokens
        totals.output_tokens += row.output_tokens
        totals.cached_input_tokens += row.cached_input_tokens
        totals.cost_usd = round(totals.cost_usd + row.cost_usd, 8)
    return CostLedger(
        wave=wave,
        updated_at=utc_now(),
        run_config_sha256=config_sha,
        dry_run=dry_run,
        enabled_model_ids=list(enabled_model_ids),
        per_model=per_model,
        totals=totals,
    )


def persist_cost_ledger(wave_dir: Path, ledger: CostLedger) -> None:
    """Atomically write ``cost.json`` with locked key order."""
    dumped = ledger.model_dump(mode="json")
    atomic_write_json(wave_dir / "cost.json", ordered_dump(dumped, COST_LEDGER_KEY_ORDER))


def require_live_keys(models: list[ModelSpec]) -> list[str]:
    """Return missing-key messages for enabled models (empty if all present)."""
    errors: list[str] = []
    for spec in models:
        message = missing_key_errors(spec.provider, spec.id)
        if message is not None:
            errors.append(message)
    return errors


def _error_transcript(
    *,
    question: Question,
    spec: ModelSpec,
    attempt: int,
    wave: str,
    config: RunConfig,
    config_sha: str,
    messages: list[RenderedMessage],
    max_output_tokens: int,
    dry_run: bool,
    started_at: str,
    error: str,
) -> Transcript:
    """Build a failed-attempt transcript so the operator can inspect and retry."""
    return Transcript(
        question_id=question.id,
        provider=spec.provider,
        requested_model=spec.id,
        response_model=None,
        attempt=attempt,
        wave=wave,
        prompt_template_id=config.prompt_template_id,
        rendered_messages=messages,
        configured_temperature=config.temperature,
        request_temperature=config.temperature if spec.send_temperature else None,
        max_output_tokens=max_output_tokens,
        run_config_sha256=config_sha,
        text="",
        raw_request={},
        raw_response={},
        usage=TokenUsage(),
        cost_usd=0.0,
        sdk_version="",
        started_at=started_at,
        finished_at=utc_now(),
        dry_run=dry_run,
        truncated=False,
        error=error,
    )


def _success_transcript(
    *,
    question: Question,
    spec: ModelSpec,
    attempt: int,
    wave: str,
    config: RunConfig,
    config_sha: str,
    messages: list[RenderedMessage],
    max_output_tokens: int,
    dry_run: bool,
    started_at: str,
    result: AdapterResult,
) -> Transcript:
    """Build a successful transcript from an adapter result."""
    return Transcript(
        question_id=question.id,
        provider=spec.provider,
        requested_model=spec.id,
        response_model=result.response_model,
        attempt=attempt,
        wave=wave,
        prompt_template_id=config.prompt_template_id,
        rendered_messages=messages,
        configured_temperature=config.temperature,
        request_temperature=result.request_temperature,
        max_output_tokens=max_output_tokens,
        run_config_sha256=config_sha,
        text=result.text,
        raw_request=result.raw_request,
        raw_response=result.raw_response,
        usage=result.usage,
        cost_usd=estimate_cost_usd(result.usage, spec.pricing_usd_per_mtok),
        sdk_version=result.sdk_version,
        started_at=started_at,
        finished_at=utc_now(),
        dry_run=dry_run,
        truncated=result.truncated,
        error=None,
        served_by=result.served_by,
    )


def run_wave(
    *,
    questions: list[Question],
    config: RunConfig,
    config_sha: str,
    models: list[ModelSpec],
    results_root: Path,
    wave_slug: str,
    dry_run: bool,
) -> int:
    """Run independent (question, model, attempt) calls. Return process exit code."""
    wave = wave_dirname(wave_slug)
    wave_dir = results_root / wave
    skipped_by_model: dict[str, int] = defaultdict(int)
    remaining_errors = 0

    for spec in models:
        adapter = adapter_for(
            spec.provider,
            dry_run=dry_run,
            openrouter_providers=spec.openrouter_providers,
        )
        max_tokens = effective_max_output_tokens(spec, config)
        for question in questions:
            messages = render_prompt(question, config)
            system_msg = next((item for item in messages if item.role == "system"), None)
            user_msg = next((item for item in messages if item.role == "user"), None)
            if system_msg is None or user_msg is None:
                raise WaveError(f"{question.id}: rendered prompt missing system or user")
            system = system_msg.content
            user = user_msg.content
            for attempt in range(1, config.attempts + 1):
                path = transcript_path(results_root, wave, spec.id, question.id, attempt)
                if load_complete_transcript(path) is not None:
                    skipped_by_model[spec.id] += 1
                    print(f"{spec.id} {question.id} {attempt} skip", file=sys.stderr)
                    continue
                started_at = utc_now()
                try:
                    result = call_with_retry(
                        adapter,
                        retry_max=config.retry_max,
                        retry_base_s=config.retry_base_s,
                        sleep=not dry_run,
                        model_id=spec.id,
                        system=system,
                        user=user,
                        temperature=config.temperature,
                        send_temperature=spec.send_temperature,
                        max_output_tokens=max_tokens,
                        timeout_s=config.timeout_s,
                    )
                    transcript = _success_transcript(
                        question=question,
                        spec=spec,
                        attempt=attempt,
                        wave=wave,
                        config=config,
                        config_sha=config_sha,
                        messages=messages,
                        max_output_tokens=max_tokens,
                        dry_run=dry_run,
                        started_at=started_at,
                        result=result,
                    )
                    status = "wrote truncated" if result.truncated else "wrote"
                except Exception as exc:  # noqa: BLE001 - persist any adapter/SDK failure
                    transcript = _error_transcript(
                        question=question,
                        spec=spec,
                        attempt=attempt,
                        wave=wave,
                        config=config,
                        config_sha=config_sha,
                        messages=messages,
                        max_output_tokens=max_tokens,
                        dry_run=dry_run,
                        started_at=started_at,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    remaining_errors += 1
                    status = "error"
                dump_transcript(transcript, path)
                ledger = rebuild_cost_ledger(
                    wave_dir=wave_dir,
                    wave=wave,
                    config_sha=config_sha,
                    dry_run=dry_run,
                    enabled_model_ids=[item.id for item in models],
                    skipped_by_model=dict(skipped_by_model),
                )
                persist_cost_ledger(wave_dir, ledger)
                print(f"{spec.id} {question.id} {attempt} {status}", file=sys.stderr)
                if not dry_run and config.sleep_between_calls_s > 0 and status.startswith("wrote"):
                    time.sleep(config.sleep_between_calls_s)

    ledger = rebuild_cost_ledger(
        wave_dir=wave_dir,
        wave=wave,
        config_sha=config_sha,
        dry_run=dry_run,
        enabled_model_ids=[item.id for item in models],
        skipped_by_model=dict(skipped_by_model),
    )
    persist_cost_ledger(wave_dir, ledger)
    return 2 if remaining_errors else 0


def verify_models(
    *,
    config: RunConfig,
    models: list[ModelSpec],
    dry_run: bool,
) -> int:
    """Ping each model with a VERIFY_MAX_TOKENS budget; print requested vs echoed id.

    A successful API response passes even when visible text is empty (reasoning
    models may spend the whole ping budget thinking): verify checks that the id
    exists and answers, not what it says. Only an exception counts as failure.
    """
    failures = 0
    for spec in models:
        adapter = adapter_for(
            spec.provider,
            dry_run=dry_run,
            openrouter_providers=spec.openrouter_providers,
        )
        try:
            result = call_with_retry(
                adapter,
                retry_max=config.retry_max,
                retry_base_s=config.retry_base_s,
                sleep=not dry_run,
                model_id=spec.id,
                system=VERIFY_SYSTEM,
                user=VERIFY_USER,
                temperature=config.temperature,
                send_temperature=spec.send_temperature,
                max_output_tokens=VERIFY_MAX_TOKENS,
                timeout_s=config.timeout_s,
            )
        except Exception as exc:  # noqa: BLE001 - ping failures become a non-zero verify exit
            print(f"{spec.id}\t requested={spec.id}\t error={exc}", file=sys.stderr)
            failures += 1
            continue
        echoed = result.response_model if result.response_model is not None else ""
        served = result.served_by if result.served_by is not None else ""
        print(f"{spec.id}\t requested={spec.id}\t echoed={echoed}\t served_by={served}")
    return 2 if failures else 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run each question against each model under identical treatment."
    )
    parser.add_argument(
        "--questions", type=Path, default=None, help="JSON array of Question objects"
    )
    parser.add_argument("--wave", default=None, help="Wave slug (1 → wave-1, smoke → wave-smoke)")
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
        help="Directory that will contain wave-N/ folders",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Write fake transcripts; no API calls"
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        default=None,
        help="Enabled model id to include (repeatable)",
    )
    parser.add_argument(
        "--verify-models",
        action="store_true",
        help="One-token ping per enabled model (mocked in tests; live needs keys)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry: validate config, optionally verify models, then run the wave."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.questions is None and not args.verify_models:
        parser.error("--questions is required unless --verify-models")
    if args.questions is not None and args.wave is None:
        parser.error("--wave is required with --questions")

    config_path = args.config if args.config is not None else default_config_path()
    try:
        config = load_run_config(config_path)
        config_sha = run_config_sha256(config_path)
        models = select_models(config, args.models)
    except (OSError, WaveError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if len(models) == 0:
        print("error: no enabled models selected", file=sys.stderr)
        return 1

    if not args.dry_run:
        missing = require_live_keys(models)
        if missing:
            for line in missing:
                print(f"error: {line}", file=sys.stderr)
            return 1

    if args.verify_models:
        verify_code = verify_models(config=config, models=models, dry_run=args.dry_run)
        if args.questions is None:
            return verify_code
        if verify_code != 0:
            return verify_code

    questions_path: Path = args.questions
    wave_slug: str = args.wave
    try:
        questions = load_questions(questions_path)
        if len(questions) == 0:
            raise WaveError("wave is empty")
        return run_wave(
            questions=questions,
            config=config,
            config_sha=config_sha,
            models=models,
            results_root=args.results_root,
            wave_slug=wave_slug,
            dry_run=args.dry_run,
        )
    except (OSError, WaveError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
