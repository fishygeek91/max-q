"""Tests for wave loading, scoring-field rules, inventory, and hashing."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

from maxq.schema import Difficulty, Domain, Question, ScoringMode
from maxq.wave import (
    HASH_ROW_RE,
    HashRowError,
    WaveError,
    append_or_verify_hash_row,
    check_inventory,
    format_hash_row,
    frozen_hash_for,
    hashes_wave_label,
    load_questions,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).parent / "fixtures" / "sample_wave.json"
PRIVATE_WAVE = ROOT / "questions" / "private" / "wave-1.json"
WAVE1_FROZEN_SHA = "f7b1f78c5e35c4b2d58472bf361d6e117980ed3bbde358d89b7229336872bc5e"

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


def test_hashes_wave_label_strips_prefix() -> None:
    """CLI --wave values normalize to the HASHES.md Wave column."""
    assert hashes_wave_label("1") == "1"
    assert hashes_wave_label("wave-1") == "1"
    assert hashes_wave_label("WAVE-1") == "1"


def test_format_hash_row_round_trip() -> None:
    """format_hash_row is parseable by HASH_ROW_RE and leaves Posted/Published open."""
    digest = "ab" * 32
    line = format_hash_row("2", "2026-09-14T12:00:00Z", digest)
    assert line == f"| 2 | 2026-09-14T12:00:00Z | {digest} | — | — |"
    match = HASH_ROW_RE.match(line)
    assert match is not None
    assert match.group("wave") == "2"
    assert match.group("frozen") == "2026-09-14T12:00:00Z"
    assert match.group("sha") == digest


def test_frozen_hash_for_committed_wave1() -> None:
    """The committed Wave 1 HASHES.md row parses to the posted digest."""
    hashes = ROOT / "questions" / "HASHES.md"
    assert frozen_hash_for("1", hashes) == WAVE1_FROZEN_SHA
    assert frozen_hash_for("99", hashes) is None


def test_append_or_verify_inserts_inside_table(tmp_path: Path) -> None:
    """A new row lands in the markdown table, before the Posted prose."""
    hashes = tmp_path / "HASHES.md"
    hashes.write_text((ROOT / "questions" / "HASHES.md").read_text(encoding="utf-8"), encoding="utf-8")
    digest = "cd" * 32
    update = append_or_verify_hash_row(
        hashes, "99", digest, "2026-09-14T12:00:00Z", write=True
    )
    assert update.status == "appended"
    text = hashes.read_text(encoding="utf-8")
    assert text.index("| 99 |") < text.index("**Posted**")
    assert frozen_hash_for("1", hashes) == WAVE1_FROZEN_SHA
    assert frozen_hash_for("99", hashes) == digest


def test_append_or_verify_is_idempotent(tmp_path: Path) -> None:
    """A second write with the same digest leaves HASHES.md unchanged."""
    hashes = tmp_path / "HASHES.md"
    hashes.write_text((ROOT / "questions" / "HASHES.md").read_text(encoding="utf-8"), encoding="utf-8")
    digest = "ef" * 32
    first = append_or_verify_hash_row(
        hashes, "88", digest, "2026-09-14T12:00:00Z", write=True
    )
    before = hashes.read_text(encoding="utf-8")
    second = append_or_verify_hash_row(
        hashes, "88", digest, "2026-09-14T13:00:00Z", write=True
    )
    assert first.status == "appended"
    assert second.status == "verified"
    assert second.row.frozen_utc == "2026-09-14T12:00:00Z"
    assert hashes.read_text(encoding="utf-8") == before


def test_append_or_verify_dry_run_does_not_write(tmp_path: Path) -> None:
    """write=False reports would_append and does not touch the file."""
    hashes = tmp_path / "HASHES.md"
    original = (ROOT / "questions" / "HASHES.md").read_text(encoding="utf-8")
    hashes.write_text(original, encoding="utf-8")
    digest = "11" * 32
    update = append_or_verify_hash_row(
        hashes, "77", digest, "2026-09-14T12:00:00Z", write=False
    )
    assert update.status == "would_append"
    assert hashes.read_text(encoding="utf-8") == original


def test_append_or_verify_refuses_mismatch(tmp_path: Path) -> None:
    """A different digest for an existing wave is a hard conflict."""
    hashes = tmp_path / "HASHES.md"
    hashes.write_text((ROOT / "questions" / "HASHES.md").read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(HashRowError, match="frozen as"):
        append_or_verify_hash_row(
            hashes, "1", "aa" * 32, "2026-09-14T12:00:00Z", write=True
        )
    assert frozen_hash_for("1", hashes) == WAVE1_FROZEN_SHA


def test_published_waves_match_hashes() -> None:
    """Published questions/wave-*.json files must match HASHES.md (none yet)."""
    hashes = ROOT / "questions" / "HASHES.md"
    published = sorted((ROOT / "questions").glob("wave-*.json"))
    for path in published:
        label = hashes_wave_label(path.stem)
        frozen = frozen_hash_for(label, hashes)
        assert frozen is not None, f"{path} has no HASHES.md row"
        assert sha256_file(path) == frozen


def _imported_names(path: Path) -> set[str]:
    """Collect import module names from ``path`` so this test is order-independent."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                names.add(alias.name)
                if module:
                    names.add(module)
                    names.add(f"{module}.{alias.name}")
    return names


def test_wave_module_does_not_import_runner_or_scoring() -> None:
    """wave.py and __init__.py must not import runner or scoring (AST, any test order)."""
    names = _imported_names(ROOT / "maxq" / "wave.py") | _imported_names(ROOT / "maxq" / "__init__.py")
    banned = {"maxq.runner", "maxq.scoring", "runner", "scoring"}
    assert names.isdisjoint(banned)


@pytest.mark.skipif(not PRIVATE_WAVE.is_file(), reason="held-out wave is local-only")
def test_private_wave_inventory_if_present() -> None:
    """If the private file exists, inventory must pass. Do not assert answers."""
    questions = load_questions(PRIVATE_WAVE)
    check_inventory(questions, require_wave1_ids=True)
    rows = [(q.id, q.domain, q.difficulty, q.scoring) for q in questions]
    assert len(rows) == 50
    hashes = ROOT / "questions" / "HASHES.md"
    frozen = frozen_hash_for("1", hashes)
    assert frozen is not None
    assert sha256_file(PRIVATE_WAVE) == frozen
