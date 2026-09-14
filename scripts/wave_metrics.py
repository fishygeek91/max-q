"""Print stem-free Wave rehearsal metrics from scored.json, cost.json, transcripts.

Usage::

    python scripts/wave_metrics.py --wave 1
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from maxq.runner import default_config_path, load_run_config, select_models, wave_dirname
from maxq.schema import CostLedger, OverrideLogLine, Transcript, WaveScoreReport
from maxq.wave import WaveError


def _parse_utc(stamp: str) -> datetime | None:
    """Parse a transcript UTC timestamp; return None if unparseable."""
    text = stamp.strip()
    if text == "":
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def format_wave_metrics(
    *,
    wave: str,
    results_root: Path,
    enabled_ids: list[str],
) -> str:
    """Return a stem-free TSV-ish report for extraction, rubric, cost, wall-clock."""
    wave_dir = results_root / wave
    scored_path = wave_dir / "scored.json"
    cost_path = wave_dir / "cost.json"
    overrides_path = wave_dir / "overrides.jsonl"
    if not scored_path.is_file():
        raise WaveError(f"{scored_path}: scored.json is missing")
    report = WaveScoreReport.model_validate(json.loads(scored_path.read_text(encoding="utf-8")))
    lines: list[str] = [
        f"wave\t{report.wave}",
        f"n_attempts\t{report.n_attempts}",
        f"enabled_models\t{','.join(enabled_ids)}",
        "",
        "model\tpass@1\tbest-of-n\ttruncated\tunparseable\tpending_rubric\terrors\tmissing",
    ]
    for summary in report.models:
        lines.append(
            "\t".join(
                [
                    summary.model,
                    f"{summary.pass_at_1:.3f}",
                    f"{summary.best_of_n:.3f}",
                    str(summary.truncated_attempts),
                    str(summary.unparseable),
                    str(summary.pending_rubric),
                    str(summary.errors),
                    str(summary.missing),
                ]
            )
        )
    n_attempts_total = len(report.attempts)
    unparseable = sum(1 for row in report.attempts if row.status == "unparseable")
    truncated = sum(1 for row in report.attempts if row.status == "truncated")
    lines.append("")
    lines.append(f"attempts_total\t{n_attempts_total}")
    lines.append(f"unparseable_total\t{unparseable}")
    lines.append(f"truncated_total\t{truncated}")
    if n_attempts_total > 0:
        parsed = n_attempts_total - unparseable
        lines.append(f"extraction_rate\t{parsed / n_attempts_total:.3f}")

    if cost_path.is_file():
        ledger = CostLedger.model_validate(json.loads(cost_path.read_text(encoding="utf-8")))
        lines.append("")
        lines.append(f"cost_usd_total\t{ledger.totals.cost_usd}")
        for row in ledger.per_model:
            lines.append(f"cost_usd\t{row.requested_model}\t{row.cost_usd}")

    started: list[datetime] = []
    finished: list[datetime] = []
    for path in sorted(wave_dir.glob("*/*-a*.json")):
        try:
            transcript = Transcript.model_validate(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        start = _parse_utc(transcript.started_at)
        end = _parse_utc(transcript.finished_at)
        if start is not None:
            started.append(start)
        if end is not None:
            finished.append(end)
    if len(started) > 0 and len(finished) > 0:
        wall = max(finished) - min(started)
        lines.append("")
        lines.append(f"wall_clock_s\t{wall.total_seconds():.1f}")
        lines.append(f"started_at_min\t{min(started).isoformat()}")
        lines.append(f"finished_at_max\t{max(finished).isoformat()}")

    if overrides_path.is_file():
        accepts = 0
        overrides = 0
        for raw in overrides_path.read_text(encoding="utf-8").splitlines():
            if raw.strip() == "":
                continue
            line = OverrideLogLine.model_validate(json.loads(raw))
            if line.action == "accept":
                accepts += 1
            elif line.action == "override":
                overrides += 1
        confirmed = accepts + overrides
        lines.append("")
        lines.append(f"rubric_accept_rows\t{accepts}")
        lines.append(f"rubric_override_rows\t{overrides}")
        if confirmed > 0:
            lines.append(f"rubric_accept_share\t{accepts / confirmed:.3f}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Print stem-free metrics for a scored wave directory."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wave", required=True, help="Wave slug (1 → wave-1)")
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
    args = parser.parse_args(argv)
    config_path = args.config if args.config is not None else default_config_path()
    try:
        config = load_run_config(config_path)
        models = select_models(config, None)
        text = format_wave_metrics(
            wave=wave_dirname(args.wave),
            results_root=args.results_root,
            enabled_ids=[spec.id for spec in models],
        )
    except (OSError, WaveError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
