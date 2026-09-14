"""Tests for scripts/render_rubric_review.py and scripts/wave_metrics.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

from maxq.runner import dump_transcript
from maxq.schema import (
    ModelSpec,
    Pricing,
    Question,
    RubricQueueItem,
    RunConfig,
    TokenUsage,
    Transcript,
)
from maxq.scoring import score_wave
from maxq.wave import dump_questions

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, filename: str) -> ModuleType:
    """Import a scripts/*.py file without installing the scripts package."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _spec(model_id: str) -> ModelSpec:
    """Minimal ModelSpec for the review renderer tests."""
    return ModelSpec(
        id=model_id,
        provider="xai",
        role="baseline",
        send_temperature=False,
        pricing_usd_per_mtok=Pricing(input=1.0, output=1.0, cached_input=None),
    )


def _config(models: list[ModelSpec], attempts: int = 1) -> RunConfig:
    """Tiny run config that does not touch live providers."""
    return RunConfig(
        temperature=0.0,
        attempts=attempts,
        max_output_tokens=128,
        timeout_s=5.0,
        retry_max=1,
        retry_base_s=0.0,
        sleep_between_calls_s=0.0,
        prompt_template_id="test-review",
        system_prompt="sys",
        user_suffix="FINAL: <answer>",
        models=models,
    )


def _transcript(*, question_id: str, model_id: str, text: str, wave: str) -> Transcript:
    """Build one dry-run transcript for the review renderer."""
    return Transcript(
        question_id=question_id,
        provider="xai",
        requested_model=model_id,
        response_model=model_id,
        attempt=1,
        wave=wave,
        prompt_template_id="test-review",
        rendered_messages=[],
        configured_temperature=0.0,
        request_temperature=0.0,
        max_output_tokens=128,
        run_config_sha256="abc",
        text=text,
        raw_request={},
        raw_response={},
        usage=TokenUsage(),
        cost_usd=0.0,
        sdk_version="test",
        started_at="2026-01-01T00:00:00Z",
        finished_at="2026-01-01T00:00:10Z",
        dry_run=True,
        truncated=False,
        error=None,
    )


def test_render_flags_self_judge_and_omits_answer_keys(tmp_path: Path) -> None:
    """SELF-JUDGE appears when judge_model equals the contestant; keys stay out."""
    judge_id = "claude-fable-5-1"
    other_id = "grok-4.6"
    question = Question(
        id="W0-REV-001",
        domain="propulsion",
        difficulty="expert",
        prompt="Dummy review stem. Do not leak the key.",
        scoring="rubric",
        answer=None,
        unit=None,
        rel_tol=None,
        rubric=["States the governing equation", "Names the assumption"],
        provenance="Synthetic review-renderer fixture.",
    )
    questions_path = tmp_path / "wave.json"
    dump_questions([question], questions_path)
    wave_dir = tmp_path / "wave-review"
    for model_id, text in (
        (judge_id, "FINAL: self-judged answer"),
        (other_id, "FINAL: other answer"),
    ):
        path = wave_dir / model_id / f"{question.id}-a1.json"
        dump_transcript(
            _transcript(
                question_id=question.id,
                model_id=model_id,
                text=text,
                wave="wave-review",
            ),
            path,
        )
    rubric = list(question.rubric) if question.rubric is not None else []
    queue = [
        RubricQueueItem(
            question_id=question.id,
            model=judge_id,
            attempt=1,
            rubric=rubric,
            llm_scores=[True, False],
            llm_notes="self",
            judge_model=judge_id,
            status="pending",
        ),
        RubricQueueItem(
            question_id=question.id,
            model=other_id,
            attempt=1,
            rubric=rubric,
            llm_scores=[True, True],
            llm_notes="other",
            judge_model=judge_id,
            status="pending",
        ),
    ]
    wave_dir.mkdir(parents=True, exist_ok=True)
    (wave_dir / "rubric-queue.json").write_text(
        json.dumps([item.model_dump(mode="json") for item in queue], indent=2) + "\n",
        encoding="utf-8",
    )
    # Tiny config so n_attempts is 1 (committed run.json uses 3).
    tiny_config = tmp_path / "run.json"
    tiny_config.write_text(
        _config([_spec(judge_id), _spec(other_id)], attempts=1).model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    module = _load_script("render_rubric_review", "render_rubric_review.py")
    output = tmp_path / "review.md"
    code = module.main(
        [
            "--questions",
            str(questions_path),
            "--wave",
            "review",
            "--results-root",
            str(tmp_path),
            "--config",
            str(tiny_config),
            "--output",
            str(output),
        ]
    )
    assert code == 0
    text = output.read_text(encoding="utf-8")
    assert "SELF-JUDGE" in text
    assert f"### {judge_id} a1 (SELF-JUDGE pending)" in text
    assert f"### {other_id} a1 (pending)" in text
    assert "Dummy review stem" in text
    assert question.provenance not in text


def test_wave_metrics_reports_extraction_without_stems(tmp_path: Path) -> None:
    """wave_metrics.py prints extraction and cost; it does not echo prompts."""
    question = Question(
        id="W0-MET-001",
        domain="propulsion",
        difficulty="undergrad",
        prompt="SECRET STEM should not appear in metrics.",
        scoring="numeric_tolerance",
        answer=1.0,
        unit="m/s",
        rel_tol=0.01,
        rubric=None,
        provenance="Synthetic metrics fixture.",
    )
    path = tmp_path / "wave-metrics" / "test-model" / f"{question.id}-a1.json"
    dump_transcript(
        _transcript(
            question_id=question.id,
            model_id="test-model",
            text="FINAL: 1.0 m/s",
            wave="wave-metrics",
        ),
        path,
    )
    score_wave(
        questions=[question],
        config=_config([_spec("test-model")], attempts=1),
        models=[_spec("test-model")],
        results_root=tmp_path,
        wave_slug="metrics",
        judge=None,
    )
    module = _load_script("wave_metrics", "wave_metrics.py")
    text = module.format_wave_metrics(
        wave="wave-metrics",
        results_root=tmp_path,
        enabled_ids=["test-model"],
    )
    assert "extraction_rate" in text
    assert "SECRET STEM" not in text
    assert "wall_clock_s" in text
