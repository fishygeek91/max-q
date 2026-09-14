"""Load, inventory-check, and hash held-out question waves.

This module must not import ``maxq.scoring`` or ``maxq.runner``. Extra Wave 1
rules live here rather than on the Pydantic ``Question`` model so scoring can
evolve separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pint
from pydantic import TypeAdapter, ValidationError

from maxq.schema import Difficulty, Domain, Question, ScoringMode

QUESTION_KEY_ORDER: tuple[str, ...] = (
    "id",
    "domain",
    "difficulty",
    "prompt",
    "scoring",
    "answer",
    "unit",
    "rel_tol",
    "rubric",
    "provenance",
)
"""JSON object key order locked for Wave 1 freeze hashing."""

WAVE1_ID_RE = re.compile(r"^W1-(PROP|ORB|STR|GNC|TEL)-0(0[1-9]|10)$")
PREFIX_TO_DOMAIN: dict[str, Domain] = {
    "PROP": "propulsion",
    "ORB": "orbital",
    "STR": "structures",
    "GNC": "gnc",
    "TEL": "telemetry",
}
SUFFIX_TO_DIFFICULTY: dict[str, Difficulty] = {
    "001": "undergrad",
    "002": "undergrad",
    "003": "undergrad",
    "004": "practitioner",
    "005": "practitioner",
    "006": "practitioner",
    "007": "practitioner",
    "008": "expert",
    "009": "expert",
    "010": "expert",
}
EXPECTED_WAVE_SIZE = 50
PER_DOMAIN = 10
TIER_COUNTS: dict[Difficulty, int] = {
    "undergrad": 3,
    "practitioner": 4,
    "expert": 3,
}
REL_TOL_MIN = 0.0
REL_TOL_MAX = 0.05
SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
HASH_ROW_RE = re.compile(
    r"^\|\s*(?P<wave>[^|]+?)\s*\|\s*(?P<frozen>[^|]+?)\s*\|\s*(?P<sha>[0-9a-f]{64})\s*\|"
)

_UREG = pint.UnitRegistry()
_QUESTIONS_ADAPTER = TypeAdapter(list[Question])


class WaveError(ValueError):
    """Raised when a wave file fails schema, scoring-field, or inventory checks."""


class HashRowError(ValueError):
    """Raised when HASHES.md is malformed or conflicts with a new digest."""


@dataclass(frozen=True)
class HashRow:
    """One data row parsed from ``questions/HASHES.md``."""

    wave: str
    frozen_utc: str
    sha256: str
    line: str


@dataclass(frozen=True)
class HashRowUpdate:
    """Outcome of comparing a digest against ``questions/HASHES.md``."""

    status: Literal["appended", "verified", "would_append"]
    row: HashRow


def sha256_file(path: Path) -> str:
    """Return the lowercase hex SHA-256 digest of ``path`` as stored on disk."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hashes_wave_label(wave: str) -> str:
    """Normalize ``--wave 1`` / ``wave-1`` to the HASHES.md Wave column value."""
    text = wave.strip()
    prefix = "wave-"
    if text.lower().startswith(prefix):
        return text[len(prefix) :]
    return text


def format_hash_row(wave_label: str, frozen_utc: str, sha256: str) -> str:
    """Return a HASHES.md table row with Posted and Published left as em dashes."""
    label = wave_label.strip()
    timestamp = frozen_utc.strip()
    digest = sha256.strip().lower()
    if not label:
        raise HashRowError("wave label is empty")
    if not timestamp:
        raise HashRowError("frozen UTC timestamp is empty")
    if SHA256_HEX_RE.fullmatch(digest) is None:
        raise HashRowError(f"invalid SHA-256 digest: {sha256!r}")
    return f"| {label} | {timestamp} | {digest} | — | — |"


def hash_row_for(wave_label: str, hashes_path: Path) -> HashRow | None:
    """Return the HASHES.md data row for ``wave_label``, if present."""
    if not hashes_path.is_file():
        return None
    return _hash_row_from_text(wave_label, hashes_path.read_text(encoding="utf-8"))


def frozen_hash_for(wave_label: str, hashes_path: Path) -> str | None:
    """Return the committed SHA-256 for ``wave_label`` from HASHES.md, if any."""
    row = hash_row_for(wave_label, hashes_path)
    if row is None:
        return None
    return row.sha256


def append_or_verify_hash_row(
    hashes_path: Path,
    wave_label: str,
    sha256: str,
    frozen_utc: str,
    *,
    write: bool,
) -> HashRowUpdate:
    """Append a HASHES.md row or verify it already matches ``sha256``.

    Never duplicates a wave. Never changes Frozen UTC, Posted, or Published on
    an existing row. Raises ``HashRowError`` if the wave is already frozen to a
    different digest.
    """
    if not hashes_path.is_file():
        raise FileNotFoundError(f"{hashes_path}: file not found")
    digest = sha256.strip().lower()
    proposed = format_hash_row(wave_label, frozen_utc, digest)
    text = hashes_path.read_text(encoding="utf-8")
    existing = _hash_row_from_text(wave_label.strip(), text)
    if existing is not None:
        if existing.sha256 == digest:
            return HashRowUpdate(status="verified", row=existing)
        raise HashRowError(
            f"wave {wave_label.strip()!r} is frozen as {existing.sha256}, file is {digest}"
        )
    new_row = HashRow(
        wave=wave_label.strip(),
        frozen_utc=frozen_utc.strip(),
        sha256=digest,
        line=proposed,
    )
    if not write:
        return HashRowUpdate(status="would_append", row=new_row)
    hashes_path.write_text(_insert_hash_row(text, proposed), encoding="utf-8", newline="\n")
    return HashRowUpdate(status="appended", row=new_row)


def _hash_row_from_text(wave_label: str, text: str) -> HashRow | None:
    """Return the first HASHES.md data row whose Wave column equals ``wave_label``."""
    for line in text.splitlines():
        parsed = _parse_hash_row(line)
        if parsed is not None and parsed.wave == wave_label:
            return parsed
    return None


def _parse_hash_row(line: str) -> HashRow | None:
    """Parse a HASHES.md data row, or return None for headers and prose."""
    match = HASH_ROW_RE.match(line.strip())
    if match is None:
        return None
    return HashRow(
        wave=match.group("wave"),
        frozen_utc=match.group("frozen").strip(),
        sha256=match.group("sha"),
        line=line.strip(),
    )


def _insert_hash_row(text: str, new_line: str) -> str:
    """Insert ``new_line`` after the last HASHES.md data row (or the separator)."""
    lines = text.splitlines()
    insert_at: int | None = None
    separator_at: int | None = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if HASH_ROW_RE.match(stripped) is not None:
            insert_at = index
        elif stripped.startswith("|---"):
            separator_at = index
    if insert_at is not None:
        lines.insert(insert_at + 1, new_line)
    elif separator_at is not None:
        lines.insert(separator_at + 1, new_line)
    else:
        raise HashRowError("HASHES.md has no markdown table to append to")
    return "\n".join(lines) + "\n"


def dump_questions(questions: list[Question], path: Path) -> None:
    """Write ``questions`` with the locked Wave 1 JSON serializer.

    UTF-8, Unix newlines, indent 2, ``ensure_ascii=False``, all schema keys
    present (including JSON ``null``), one trailing newline. Do not re-serialize
    a frozen file.
    """
    payload = [_ordered_question_dict(question) for question in questions]
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def load_questions(path: Path) -> list[Question]:
    """Load a JSON array of ``Question`` objects from ``path``.

    Raises:
        WaveError: Root JSON is not an array, or Pydantic validation fails.
    """
    raw_text = path.read_text(encoding="utf-8")
    try:
        payload: object = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise WaveError(f"{path}: invalid JSON ({exc})") from exc
    if not isinstance(payload, list):
        raise WaveError(f"{path}: root must be a JSON array, got {type(payload).__name__}")
    try:
        questions = _QUESTIONS_ADAPTER.validate_python(payload)
    except ValidationError as exc:
        raise WaveError(f"{path}: schema validation failed:\n{exc}") from exc
    return questions


def check_scoring_fields(question: Question) -> None:
    """Enforce per-scoring-mode field rules that the Pydantic model does not."""
    if not question.provenance.strip():
        raise WaveError(f"{question.id}: provenance must be non-empty")
    if not question.prompt.strip():
        raise WaveError(f"{question.id}: prompt must be non-empty")

    mode: ScoringMode = question.scoring
    if mode == "numeric_tolerance":
        _require_numeric_answer(question)
        _require_pint_unit(question)
        _require_rel_tol(question)
        if question.rubric is not None:
            raise WaveError(f"{question.id}: numeric_tolerance must have rubric=null")
        return
    if mode == "exact":
        if not isinstance(question.answer, str) or not question.answer.strip():
            raise WaveError(f"{question.id}: exact scoring requires a non-empty string answer")
        if question.unit is not None or question.rel_tol is not None or question.rubric is not None:
            raise WaveError(f"{question.id}: exact scoring requires unit, rel_tol, and rubric null")
        return
    if mode == "rubric":
        if question.rubric is None or len(question.rubric) == 0:
            raise WaveError(f"{question.id}: rubric scoring requires a non-empty rubric list")
        if any(not item.strip() for item in question.rubric):
            raise WaveError(f"{question.id}: rubric bullets must be non-empty strings")
        return
    if mode == "ground_truth_series":
        _require_series_answer(question)
        _require_pint_unit(question)
        _require_rel_tol(question)
        if question.rubric is not None:
            raise WaveError(f"{question.id}: ground_truth_series must have rubric=null")
        return
    raise WaveError(f"{question.id}: unsupported scoring mode {mode!r}")


def check_inventory(questions: list[Question], *, require_wave1_ids: bool = True) -> None:
    """Check unique IDs, scoring fields, and (for a complete wave) 10×3/4/3 mix."""
    if len(questions) == 0:
        raise WaveError("wave is empty")

    ids = [question.id for question in questions]
    duplicates = [qid for qid, count in Counter(ids).items() if count > 1]
    if duplicates:
        raise WaveError(f"duplicate ids: {duplicates}")

    for question in questions:
        check_scoring_fields(question)
        if require_wave1_ids:
            _check_wave1_id(question)

    if len(questions) != EXPECTED_WAVE_SIZE:
        return

    domains = Counter(question.domain for question in questions)
    for domain in PREFIX_TO_DOMAIN.values():
        if domains[domain] != PER_DOMAIN:
            raise WaveError(f"domain {domain} has {domains[domain]} items, expected {PER_DOMAIN}")

    by_domain: dict[Domain, list[Question]] = {domain: [] for domain in PREFIX_TO_DOMAIN.values()}
    for question in questions:
        by_domain[question.domain].append(question)
    for domain, group in by_domain.items():
        tiers = Counter(item.difficulty for item in group)
        for tier, expected in TIER_COUNTS.items():
            if tiers[tier] != expected:
                raise WaveError(
                    f"{domain} has {tiers[tier]} {tier} items, expected {expected}"
                )


def inventory_rows(questions: list[Question]) -> list[tuple[str, str, str, str]]:
    """Return ``(id, domain, difficulty, scoring)`` rows with no stems or answers."""
    return [
        (question.id, question.domain, question.difficulty, question.scoring)
        for question in questions
    ]


def _ordered_question_dict(question: Question) -> dict[str, object]:
    dumped: Mapping[str, object] = question.model_dump()
    return {key: dumped[key] for key in QUESTION_KEY_ORDER}


def _check_wave1_id(question: Question) -> None:
    match = WAVE1_ID_RE.fullmatch(question.id)
    if match is None:
        raise WaveError(f"{question.id}: id must match W1-<PREFIX>-001..010")
    prefix = match.group(1)
    suffix = question.id[-3:]
    expected_domain = PREFIX_TO_DOMAIN[prefix]
    if question.domain != expected_domain:
        raise WaveError(
            f"{question.id}: domain {question.domain!r} does not match prefix {prefix}"
        )
    expected_difficulty = SUFFIX_TO_DIFFICULTY[suffix]
    if question.difficulty != expected_difficulty:
        raise WaveError(
            f"{question.id}: difficulty {question.difficulty!r} does not match suffix {suffix}"
        )


def _require_numeric_answer(question: Question) -> None:
    if isinstance(question.answer, bool) or not isinstance(question.answer, (int, float)):
        raise WaveError(f"{question.id}: numeric_tolerance requires a float answer")


def _require_rel_tol(question: Question) -> None:
    if question.rel_tol is None:
        raise WaveError(f"{question.id}: missing rel_tol")
    if not (REL_TOL_MIN < question.rel_tol <= REL_TOL_MAX):
        raise WaveError(
            f"{question.id}: rel_tol {question.rel_tol} not in ({REL_TOL_MIN}, {REL_TOL_MAX}]"
        )


def _require_pint_unit(question: Question) -> None:
    if question.unit is None or not question.unit.strip():
        raise WaveError(f"{question.id}: missing pint-parseable unit")
    try:
        _UREG.parse_units(question.unit)
    except pint.UndefinedUnitError as exc:
        raise WaveError(f"{question.id}: unit {question.unit!r} is not pint-parseable") from exc


def _require_series_answer(question: Question) -> None:
    if not isinstance(question.answer, str):
        raise WaveError(f"{question.id}: ground_truth_series answer must be a JSON array string")
    try:
        parsed: object = json.loads(question.answer)
    except json.JSONDecodeError as exc:
        raise WaveError(f"{question.id}: series answer is not valid JSON") from exc
    if not isinstance(parsed, list) or len(parsed) == 0:
        raise WaveError(f"{question.id}: series answer must be a non-empty JSON array")
    for index, item in enumerate(parsed):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise WaveError(f"{question.id}: series element {index} is not numeric")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a question-wave JSON array and print a stem-free inventory."
    )
    parser.add_argument("path", type=Path, help="Path to a JSON array of Question objects")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI: validate ``path``, print inventory (no answers), print SHA-256. Exit 1 on failure."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    path: Path = args.path
    try:
        questions = load_questions(path)
        require_ids = all(WAVE1_ID_RE.fullmatch(question.id) for question in questions)
        check_inventory(questions, require_wave1_ids=require_ids)
    except (OSError, WaveError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    digest = sha256_file(path)
    print(f"file: {path}")
    print(f"count: {len(questions)}")
    print(f"sha256: {digest}")
    print(f"bytes: {path.stat().st_size}")
    print("id\tdomain\tdifficulty\tscoring")
    for row in inventory_rows(questions):
        print("\t".join(row))
    if len(questions) != EXPECTED_WAVE_SIZE:
        print(f"note: incomplete wave ({len(questions)}/{EXPECTED_WAVE_SIZE})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
