"""Render grouped-by-question markdown for human rubric confirmation.

Output includes stems and model answers, so it must stay under gitignored
``results/wave-*/``. Do not print question answer keys.

Usage::

    python scripts/render_rubric_review.py --wave 1 --questions questions/private/wave-1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from maxq.runner import (
    default_config_path,
    load_run_config,
    select_models,
    transcript_path,
    wave_dirname,
)
from maxq.schema import Question, RubricQueueItem, Transcript
from maxq.scoring import extract_final, load_transcript_any
from maxq.wave import WaveError, load_questions


def _load_queue(path: Path) -> dict[tuple[str, str, int], RubricQueueItem]:
    """Load ``rubric-queue.json``; missing file is an empty queue."""
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
        item = RubricQueueItem.model_validate(entry)
        items[(item.question_id, item.model, item.attempt)] = item
    return items


def _fence(text: str) -> str:
    """Wrap ``text`` in a markdown fence; neutralize inner triple-backticks."""
    cleaned = text.replace("```", "'''")
    return f"```\n{cleaned}\n```"


def _bool_mark(value: bool) -> str:
    """Return a checkbox glyph for one rubric bullet."""
    return "[x]" if value else "[ ]"


def render_rubric_review(
    *,
    questions: list[Question],
    model_ids: list[str],
    results_root: Path,
    wave_slug: str,
    n_attempts: int,
) -> str:
    """Return markdown grouped by question, then model, then attempt.

    Rows where ``judge_model`` equals the contestant ``model`` are flagged
    ``SELF-JUDGE`` so a human reviewer can scrutinize them extra.
    """
    if len(questions) == 0:
        raise WaveError("wave is empty")
    if len(model_ids) == 0:
        raise WaveError("no models selected")
    if n_attempts < 1:
        raise WaveError("n_attempts must be >= 1")
    wave = wave_dirname(wave_slug)
    queue = _load_queue(results_root / wave / "rubric-queue.json")
    rubric_questions = [question for question in questions if question.scoring == "rubric"]
    lines: list[str] = [
        f"# Rubric review ({wave})",
        "",
        (
            "Official scores stay pending until a human confirms. Self-judge rows "
            "(judge_model equals the contestant) are flagged in the attempt heading."
        ),
        "",
        "Do not commit this file: it contains held-out stems.",
        "",
    ]
    for question in rubric_questions:
        rubric = question.rubric if question.rubric is not None else []
        lines.append(f"## {question.id}")
        lines.append("")
        lines.append(f"- domain: {question.domain}")
        lines.append(f"- difficulty: {question.difficulty}")
        lines.append("")
        lines.append("### Stem")
        lines.append("")
        lines.append(_fence(question.prompt))
        lines.append("")
        lines.append("### Rubric")
        lines.append("")
        for index, bullet in enumerate(rubric):
            lines.append(f"{index}. {bullet}")
        lines.append("")
        for model_id in model_ids:
            for attempt in range(1, n_attempts + 1):
                key = (question.id, model_id, attempt)
                item = queue.get(key)
                path = transcript_path(results_root, wave, model_id, question.id, attempt)
                transcript: Transcript | None = load_transcript_any(path)
                flags: list[str] = []
                judge_model = item.judge_model if item is not None else None
                if item is not None and judge_model is not None and judge_model == model_id:
                    flags.append("SELF-JUDGE")
                if item is not None and item.status == "confirmed":
                    flags.append("confirmed")
                elif item is None:
                    flags.append("missing-queue")
                else:
                    flags.append("pending")
                flag_text = " ".join(flags)
                lines.append(f"### {model_id} a{attempt} ({flag_text})")
                lines.append("")
                if transcript is None:
                    lines.append(f"Transcript missing: `{path}`.")
                    lines.append("")
                    continue
                extracted = extract_final(transcript.text)
                lines.append(f"- extracted FINAL: {extracted if extracted is not None else '(none)'}")
                if transcript.error is not None:
                    lines.append(f"- transcript error: {transcript.error}")
                if transcript.truncated:
                    lines.append("- truncated: true")
                lines.append("")
                lines.append("#### Model answer")
                lines.append("")
                lines.append(_fence(transcript.text))
                lines.append("")
                if item is None:
                    lines.append("No rubric-queue row for this attempt.")
                    lines.append("")
                    continue
                lines.append(f"- judge_model: {item.judge_model}")
                lines.append(f"- llm_notes: {item.llm_notes}")
                lines.append("")
                scores = item.llm_scores
                if scores is None:
                    lines.append("No first-pass llm_scores yet.")
                    lines.append("")
                    continue
                lines.append("#### Judge bullets")
                lines.append("")
                for index, bullet in enumerate(rubric):
                    mark = _bool_mark(scores[index]) if index < len(scores) else "[?]"
                    lines.append(f"- {mark} {index}. {bullet}")
                lines.append("")
    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    """CLI flags for the review renderer."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wave", required=True, help="Wave slug (1 → wave-1)")
    parser.add_argument("--questions", type=Path, required=True, help="Frozen private wave file")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Run config JSON (default: repo config/run.json)",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=ROOT / "results",
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
        "--output",
        type=Path,
        default=None,
        help="Markdown path (default: results/wave-N/rubric-review.md)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Write grouped rubric-review markdown and print the output path."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    config_path = args.config if args.config is not None else default_config_path()
    try:
        config = load_run_config(config_path)
        models = select_models(config, args.models)
        questions = load_questions(args.questions)
        wave = wave_dirname(args.wave)
        markdown = render_rubric_review(
            questions=questions,
            model_ids=[spec.id for spec in models],
            results_root=args.results_root,
            wave_slug=args.wave,
            n_attempts=config.attempts,
        )
    except (OSError, WaveError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    output = args.output if args.output is not None else args.results_root / wave / "rubric-review.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8", newline="\n")
    print(str(output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
