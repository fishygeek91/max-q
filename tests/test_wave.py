"""Tests for wave loading, scoring-field rules, inventory, and hashing."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from maxq.schema import Difficulty, Domain, Question, ScoringMode
from maxq.wave import WaveError, check_inventory, load_questions, sha256_file

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).parent / "fixtures" / "sample_wave.json"
PRIVATE_WAVE = ROOT / "questions" / "private" / "wave-1.json"

DOMAINS: tuple[Domain, ...] = (
    "propulsion",
    "orbital",
    "structures",
    "gnc",
    "telemetry",
)
PREFIX = {
    "propulsion": "PROP",
    "orbital": "ORB",
    "structures": "STR",
    "gnc": "GNC",
    "telemetry": "TEL",
}
DIFF_BY_N: dict[int, Difficulty] = {
    1: "undergrad",
    2: "undergrad",
    3: "undergrad",
    4: "practitioner",
    5: "practitioner",
    6: "practitioner",
    7: "practitioner",
    8: "expert",
    9: "expert",
    10: "expert",
}


def _dummy_question(
    *,
    domain: Domain,
    index: int,
    scoring: ScoringMode = "numeric_tolerance",
    qid: str | None = None,
    difficulty: Difficulty | None = None,
    answer: str | float | None = 1.0,
    unit: str | None = "m/s",
    rel_tol: float | None = 0.01,
    rubric: list[str] | None = None,
) -> Question:
    """Build a schema-valid dummy question for inventory tests."""
    prefix = PREFIX[domain]
    suffix = f"{index:03d}"
    resolved_id = qid if qid is not None else f"W1-{prefix}-{suffix}"
    resolved_diff = difficulty if difficulty is not None else DIFF_BY_N[index]
    return Question(
        id=resolved_id,
        domain=domain,
        difficulty=resolved_diff,
        prompt="Dummy prompt. Report a single number in m/s.",
        scoring=scoring,
        answer=answer,
        unit=unit,
        rel_tol=rel_tol,
        rubric=rubric,
        provenance="Synthetic inventory fixture; not a Wave 1 item.",
    )


def _full_wave() -> list[Question]:
    """Return 50 dummy items with a legal Wave 1 mix."""
    questions: list[Question] = []
    for domain in DOMAINS:
        for index in range(1, 11):
            questions.append(_dummy_question(domain=domain, index=index))
    return questions


def test_load_valid_fixture() -> None:
    """A tiny dummy array loads as Question objects."""
    questions = load_questions(FIXTURE)
    assert len(questions) == 3
    assert questions[0].id == "W0-TEST-001"
    assert questions[1].scoring == "rubric"
    assert questions[2].scoring == "exact"


def test_reject_non_array(tmp_path: Path) -> None:
    """A JSON object root is rejected."""
    path = tmp_path / "wrapped.json"
    path.write_text(json.dumps({"questions": []}) + "\n", encoding="utf-8")
    with pytest.raises(WaveError, match="JSON array"):
        load_questions(path)


def test_reject_bad_domain(tmp_path: Path) -> None:
    """An unknown domain fails Pydantic validation inside load_questions."""
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    payload[0]["domain"] = "avionics"
    path = tmp_path / "bad_domain.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(WaveError, match="schema validation failed"):
        load_questions(path)


def test_reject_numeric_missing_unit() -> None:
    """numeric_tolerance without a unit fails scoring-field checks."""
    question = _dummy_question(domain="propulsion", index=1, unit=None)
    with pytest.raises(WaveError, match="missing pint-parseable unit"):
        check_inventory([question], require_wave1_ids=True)


def test_reject_duplicate_id() -> None:
    """Duplicate IDs fail inventory."""
    first = _dummy_question(domain="propulsion", index=1)
    second = _dummy_question(domain="propulsion", index=1)
    with pytest.raises(WaveError, match="duplicate ids"):
        check_inventory([first, second], require_wave1_ids=True)


def test_inventory_accepts_legal_wave1_mix() -> None:
    """A constructed 50-item 10×3/4/3 mix passes."""
    check_inventory(_full_wave(), require_wave1_ids=True)


def test_inventory_rejects_wrong_mix() -> None:
    """A 50-item wave with 11 propulsion items fails domain counts."""
    questions = _full_wave()
    questions[-1] = _dummy_question(domain="propulsion", index=10, qid="W1-PROP-010b")
    with pytest.raises(WaveError, match="propulsion"):
        check_inventory(questions, require_wave1_ids=False)


def test_sha256_known_bytes(tmp_path: Path) -> None:
    """sha256_file matches hashlib over the same bytes."""
    path = tmp_path / "blob.bin"
    payload = b"max-q-wave-hash-fixture\n"
    path.write_bytes(payload)
    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_wave_module_does_not_import_stubs() -> None:
    """Importing maxq.wave must not import the unimplemented scoring/runner stubs."""
    for name in ("maxq.scoring", "maxq.runner"):
        assert name not in sys.modules


@pytest.mark.skipif(not PRIVATE_WAVE.is_file(), reason="held-out wave is local-only")
def test_private_wave_inventory_if_present() -> None:
    """If the private file exists, inventory must pass. Do not assert answers."""
    questions = load_questions(PRIVATE_WAVE)
    check_inventory(questions, require_wave1_ids=True)
    rows = [(q.id, q.domain, q.difficulty, q.scoring) for q in questions]
    assert len(rows) == 50
