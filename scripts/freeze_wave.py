"""Record a wave hash commitment in questions/HASHES.md before any model run.

Hashes the question file as stored on disk (SHA-256 of exact bytes). Does not
re-serialize, pretty-print, or sort keys. Authoring may call ``dump_questions``
once to lock JSON layout; this script must never rewrite a frozen file.

Default is a dry run. Pass ``--run`` to append a HASHES.md row. Re-running on
an already-frozen wave is safe: matching digest exits 0; a mismatch exits 2.

Usage::

    python scripts/freeze_wave.py --wave 1 --questions questions/private/wave-1.json
    python scripts/freeze_wave.py --wave 1 --questions questions/private/wave-1.json --run
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from maxq.wave import (
    EXPECTED_WAVE_SIZE,
    WAVE1_ID_RE,
    HashRowError,
    WaveError,
    append_or_verify_hash_row,
    check_inventory,
    hashes_wave_label,
    load_questions,
    sha256_file,
)


def _post_template(*, wave: str, digest: str, frozen_utc: str, size: int, questions: Path) -> str:
    """Return copy-paste text for a gist, X post, or GitHub issue comment."""
    relative = questions.as_posix()
    return (
        f"## Wave {wave} hash commitment (public record)\n"
        "\n"
        f"**Benchmark:** Max-Q Wave {wave} — held-out aerospace engineering "
        "questions, frozen before any model run.\n"
        "\n"
        f"**SHA-256 (of `{relative}` exactly as stored on disk, UTF-8, "
        f"{size} bytes):**\n"
        "\n"
        "```\n"
        f"{digest}\n"
        "```\n"
        "\n"
        f"**Frozen (UTC):** {frozen_utc}\n"
        "\n"
        f"The question file will be published to `questions/wave-{wave}.json` "
        "after every enabled model has been run and scored; anyone can then "
        f"verify `sha256sum questions/wave-{wave}.json` against this post.\n"
    )


def main(argv: list[str] | None = None) -> int:
    """Validate a private wave, hash exact bytes, and optionally write HASHES.md."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wave", required=True, help="Wave slug (1 → HASHES.md column 1)")
    parser.add_argument("--questions", type=Path, required=True, help="Private wave JSON")
    parser.add_argument(
        "--hashes",
        type=Path,
        default=ROOT / "questions" / "HASHES.md",
        help="HASHES.md path (default: questions/HASHES.md)",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Append HASHES.md if this wave has no row (default: print only)",
    )
    args = parser.parse_args(argv)

    wave = hashes_wave_label(args.wave)
    if not wave:
        print("error: --wave is empty", file=sys.stderr)
        return 1

    # Step 1: schema + inventory. Never rewrite the question file.
    try:
        questions = load_questions(args.questions)
        require_ids = all(WAVE1_ID_RE.fullmatch(question.id) for question in questions)
        check_inventory(questions, require_wave1_ids=require_ids)
    except (OSError, WaveError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    # Step 2: SHA-256 of the bytes on disk — not a re-serialized canonical form.
    digest = sha256_file(args.questions)
    size = args.questions.stat().st_size
    frozen_utc = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Step 3: append, verify an existing matching row, or refuse a conflict.
    try:
        update = append_or_verify_hash_row(
            args.hashes,
            wave,
            digest,
            frozen_utc,
            write=args.run,
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except HashRowError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2

    print(f"file: {args.questions}")
    print(f"count: {len(questions)}")
    print(f"bytes: {size}")
    print(f"sha256: {digest}")
    print(f"HASHES.md: {update.status}")
    print(f"row: {update.row.line}")
    print("--- public post (copy/paste) ---")
    print(
        _post_template(
            wave=wave,
            digest=digest,
            frozen_utc=update.row.frozen_utc,
            size=size,
            questions=args.questions,
        ),
        end="",
    )
    print("---")
    if len(questions) != EXPECTED_WAVE_SIZE:
        print(
            f"note: incomplete wave ({len(questions)}/{EXPECTED_WAVE_SIZE})",
            file=sys.stderr,
        )
    if update.status == "verified":
        print("already frozen; HASHES.md row left unchanged.")
    elif not args.run:
        print("(dry run — pass --run to append this row to HASHES.md)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
