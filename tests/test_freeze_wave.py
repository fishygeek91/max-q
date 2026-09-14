"""Tests for scripts/freeze_wave.py (issue #4)."""

from __future__ import annotations

import ast
import importlib.util
import shutil
from pathlib import Path
from types import ModuleType

from maxq.wave import frozen_hash_for, sha256_file

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).parent / "fixtures" / "sample_wave.json"
HASHES = ROOT / "questions" / "HASHES.md"


def _load_freeze() -> ModuleType:
    """Import freeze_wave.py as a module without installing scripts/."""
    spec = importlib.util.spec_from_file_location(
        "freeze_wave", ROOT / "scripts" / "freeze_wave.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_freeze_script_does_not_call_dump_questions() -> None:
    """Freeze must hash on-disk bytes and must not re-serialize the wave."""
    tree = ast.parse((ROOT / "scripts" / "freeze_wave.py").read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)
    assert "dump_questions" not in names


def test_freeze_dry_run_does_not_write(tmp_path: Path) -> None:
    """Default invocation prints a row and leaves HASHES.md and the JSON untouched."""
    hashes = tmp_path / "HASHES.md"
    shutil.copyfile(HASHES, hashes)
    questions = tmp_path / "wave-77.json"
    shutil.copyfile(FIXTURE, questions)
    before_hashes = hashes.read_bytes()
    before_questions = questions.read_bytes()
    module = _load_freeze()
    code = module.main(
        [
            "--wave",
            "77",
            "--questions",
            str(questions),
            "--hashes",
            str(hashes),
        ]
    )
    assert code == 0
    assert hashes.read_bytes() == before_hashes
    assert questions.read_bytes() == before_questions
    assert frozen_hash_for("77", hashes) is None


def test_freeze_run_appends_once_and_is_idempotent(tmp_path: Path) -> None:
    """--run appends one row; a second --run verifies without duplicating."""
    hashes = tmp_path / "HASHES.md"
    shutil.copyfile(HASHES, hashes)
    questions = tmp_path / "wave-77.json"
    shutil.copyfile(FIXTURE, questions)
    digest = sha256_file(questions)
    module = _load_freeze()
    argv = [
        "--wave",
        "77",
        "--questions",
        str(questions),
        "--hashes",
        str(hashes),
        "--run",
    ]
    assert module.main(argv) == 0
    assert frozen_hash_for("77", hashes) == digest
    after_first = hashes.read_text(encoding="utf-8")
    assert after_first.count(digest) == 1
    assert module.main(argv) == 0
    after_second = hashes.read_text(encoding="utf-8")
    assert after_second == after_first
    assert questions.read_bytes() == FIXTURE.read_bytes()


def test_freeze_refuses_hash_mismatch(tmp_path: Path) -> None:
    """Changing the question file after freeze exits 2 and does not rewrite HASHES.md."""
    hashes = tmp_path / "HASHES.md"
    shutil.copyfile(HASHES, hashes)
    questions = tmp_path / "wave-77.json"
    shutil.copyfile(FIXTURE, questions)
    module = _load_freeze()
    argv = [
        "--wave",
        "77",
        "--questions",
        str(questions),
        "--hashes",
        str(hashes),
        "--run",
    ]
    assert module.main(argv) == 0
    committed = hashes.read_bytes()
    questions.write_bytes(questions.read_bytes() + b"\n")
    assert module.main(argv) == 2
    assert hashes.read_bytes() == committed


def test_freeze_wave1_match_is_already_frozen(tmp_path: Path) -> None:
    """Re-freezing a wave whose HASHES.md digest already matches is a no-op."""
    questions = tmp_path / "wave-1.json"
    shutil.copyfile(FIXTURE, questions)
    hashes = tmp_path / "HASHES.md"
    hashes.write_text(
        (
            "# hashes\n\n"
            "| Wave | Frozen (UTC) | SHA-256 | Posted | Published |\n"
            "|---|---|---|---|---|\n"
            f"| 1 | 2026-09-13T14:38:08Z | {sha256_file(questions)} | — | — |\n\n"
            "**Posted** = external record.\n"
        ),
        encoding="utf-8",
    )
    before = hashes.read_text(encoding="utf-8")
    before_questions = questions.read_bytes()
    module = _load_freeze()
    code = module.main(
        [
            "--wave",
            "1",
            "--questions",
            str(questions),
            "--hashes",
            str(hashes),
            "--run",
        ]
    )
    assert code == 0
    assert hashes.read_text(encoding="utf-8") == before
    assert questions.read_bytes() == before_questions
