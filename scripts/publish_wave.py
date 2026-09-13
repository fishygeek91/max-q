"""Publish a completed wave: guard, then force-add transcripts + questions.

``results/wave-*/`` is gitignored because transcripts embed held-out question
stems (issue #10). Publishing early — in particular after the Grok 4.6 baseline
run but before a later Grok generation has been scored — leaks the wave and
voids the delta measurement. This script is the only sanctioned path to
publication:

1. Refuse unless EVERY enabled model in the run config has a complete,
   error-free transcript for every (question, attempt) pair.
2. Verify the question file still matches its frozen SHA-256 in
   ``questions/HASHES.md``.
3. Print (or with ``--run`` execute) the exact ``git add -f`` commands.

Usage::

    python scripts/publish_wave.py --wave 1 --questions questions/private/wave-1.json
    python scripts/publish_wave.py --wave 1 --questions questions/private/wave-1.json --run
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from maxq.runner import (
    default_config_path,
    load_complete_transcript,
    load_run_config,
    transcript_path,
    wave_dirname,
)
from maxq.wave import WaveError, load_questions, sha256_file

HASH_ROW_RE = re.compile(r"^\|\s*(?P<wave>[^|]+?)\s*\|[^|]*\|\s*(?P<sha>[0-9a-f]{64})\s*\|")


def frozen_hash_for(wave_label: str, hashes_path: Path) -> str | None:
    """Return the committed SHA-256 for ``wave_label`` from HASHES.md, if any."""
    if not hashes_path.is_file():
        return None
    for line in hashes_path.read_text(encoding="utf-8").splitlines():
        match = HASH_ROW_RE.match(line.strip())
        if match is not None and match.group("wave") == wave_label:
            return match.group("sha")
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wave", required=True, help="Wave slug (1 → wave-1)")
    parser.add_argument("--questions", type=Path, required=True, help="Frozen private wave file")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    parser.add_argument("--run", action="store_true", help="Execute git add -f (default: print)")
    args = parser.parse_args(argv)

    config_path = args.config if args.config is not None else default_config_path()
    try:
        config = load_run_config(config_path)
        questions = load_questions(args.questions)
        wave = wave_dirname(args.wave)
    except (OSError, WaveError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    enabled = [spec for spec in config.models if spec.enabled]
    missing: list[str] = []
    truncated = 0
    for spec in enabled:
        for question in questions:
            for attempt in range(1, config.attempts + 1):
                path = transcript_path(args.results_root, wave, spec.id, question.id, attempt)
                transcript = load_complete_transcript(path)
                if transcript is None:
                    missing.append(f"{spec.id} {question.id} attempt {attempt}")
                elif transcript.truncated:
                    truncated += 1
    if missing:
        print(
            f"REFUSED: {len(missing)} incomplete (question, model, attempt) triples.\n"
            "Every ENABLED model in config/run.json must be fully run and error-free\n"
            "before the wave is published (disable a model row only with a written\n"
            "rationale in its notes). First missing:",
            file=sys.stderr,
        )
        for line in missing[:10]:
            print(f"  {line}", file=sys.stderr)
        return 2
    if truncated:
        print(f"note: {truncated} transcripts are truncated (token cap); publishing anyway.")

    stripped = wave.removeprefix("wave-")
    frozen = frozen_hash_for(stripped, ROOT / "questions" / "HASHES.md")
    actual = sha256_file(args.questions)
    if frozen is None:
        print(f"error: no HASHES.md row for wave {stripped!r}", file=sys.stderr)
        return 1
    if frozen != actual:
        print(
            f"REFUSED: {args.questions} sha256 {actual} does not match frozen {frozen}",
            file=sys.stderr,
        )
        return 2

    publish_target = ROOT / "questions" / f"{wave}.json"
    if not publish_target.exists():
        shutil.copyfile(args.questions, publish_target)
        print(f"copied {args.questions} -> {publish_target}")
    if sha256_file(publish_target) != frozen:
        print(f"REFUSED: {publish_target} does not match the frozen hash", file=sys.stderr)
        return 2

    commands = [
        ["git", "add", "-f", str((args.results_root / wave).relative_to(ROOT))],
        ["git", "add", str(publish_target.relative_to(ROOT))],
    ]
    for command in commands:
        line = " ".join(command)
        if args.run:
            subprocess.run(command, cwd=ROOT, check=True)
            print(f"ran: {line}")
        else:
            print(line)
    if not args.run:
        print(
            "(dry run — pass --run to execute, then commit and update HASHES.md Published column)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
