"""Tests for unit-aware scoring, rubric queue, and pass@1 / best-of-n (issue #3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from maxq.runner import dump_transcript, load_run_config
from maxq.schema import (
    AdapterResult,
    ModelSpec,
    Pricing,
    Question,
    RubricAcceptSpec,
    RubricOverrideSpec,
    RunConfig,
    TokenUsage,
    Transcript,
)
from maxq.scoring import (
    ProviderRubricJudge,
    RubricJudgment,
    StubRubricJudge,
    extract_final,
    format_score_table,
    main,
    parse_judge_json,
    parse_numeric,
    score_attempt,
    score_exact,
    score_numeric,
    score_series,
    score_wave,
    within_rel_tol,
)
from maxq.wave import WaveError, load_questions

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).parent / "fixtures" / "scoring_wave.json"
CONFIG = ROOT / "config" / "run.json"
MODEL_ID = "test-model"


class FakeJudge:
    """Deterministic rubric judge for tests. No API calls."""

    def __init__(self, scores: list[bool], notes: str = "fake judge") -> None:
        self.scores = scores
        self.notes = notes
        self.calls = 0

    def score(
        self,
        *,
        prompt: str,
        transcript_text: str,
        rubric: list[str],
    ) -> RubricJudgment:
        """Return the canned per-bullet scores after checking length."""
        del prompt, transcript_text
        if len(self.scores) != len(rubric):
            raise ValueError("fake judge score length does not match rubric")
        self.calls += 1
        return RubricJudgment(criterion_scores=list(self.scores), notes=self.notes)


class FakeAdapter:
    """Provider adapter that returns canned visible text."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    def complete(
        self,
        *,
        model_id: str,
        system: str,
        user: str,
        temperature: float,
        send_temperature: bool,
        max_output_tokens: int,
        timeout_s: float,
    ) -> AdapterResult:
        """Ignore the prompt and return ``self.text``."""
        del system, user, temperature, send_temperature, max_output_tokens, timeout_s
        self.calls += 1
        return AdapterResult(
            text=self.text,
            usage=TokenUsage(),
            raw_request={"model": model_id},
            raw_response={"text": self.text},
            request_temperature=0.0,
            response_model=model_id,
            sdk_version="test",
        )


def _spec(model_id: str = MODEL_ID) -> ModelSpec:
    """Return a minimal ModelSpec for in-process score_wave tests."""
    return ModelSpec(
        id=model_id,
        provider="xai",
        role="baseline",
        send_temperature=False,
        pricing_usd_per_mtok=Pricing(input=1.0, output=1.0, cached_input=None),
    )


def _config(attempts: int = 2) -> RunConfig:
    """Return a tiny run config that does not touch live providers."""
    return RunConfig(
        temperature=0.0,
        attempts=attempts,
        max_output_tokens=128,
        timeout_s=5.0,
        retry_max=1,
        retry_base_s=0.0,
        sleep_between_calls_s=0.0,
        prompt_template_id="test-scoring",
        system_prompt="You are scoring tests.",
        user_suffix="Put the final answer on the last line as FINAL: <answer>",
        models=[_spec()],
    )


def _question(**overrides: object) -> Question:
    """Build a numeric undergrad question, with field overrides."""
    payload: dict[str, object] = {
        "id": "W0-NUM-001",
        "domain": "propulsion",
        "difficulty": "undergrad",
        "prompt": "Report a speed in m/s.",
        "scoring": "numeric_tolerance",
        "answer": 1.0,
        "unit": "m/s",
        "rel_tol": 0.01,
        "rubric": None,
        "provenance": "Synthetic scoring unit test.",
    }
    payload.update(overrides)
    return Question.model_validate(payload)


def _transcript(**overrides: object) -> Transcript:
    """Build a dry-run transcript with sensible defaults."""
    payload: dict[str, object] = {
        "question_id": "W0-NUM-001",
        "provider": "xai",
        "requested_model": MODEL_ID,
        "response_model": MODEL_ID,
        "attempt": 1,
        "wave": "wave-score",
        "prompt_template_id": "test-scoring",
        "rendered_messages": [],
        "configured_temperature": 0.0,
        "request_temperature": 0.0,
        "max_output_tokens": 128,
        "run_config_sha256": "abc",
        "text": "FINAL: 1.0 m/s",
        "raw_request": {},
        "raw_response": {},
        "usage": {"input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0},
        "cost_usd": 0.0,
        "sdk_version": "test",
        "started_at": "2026-01-01T00:00:00Z",
        "finished_at": "2026-01-01T00:00:01Z",
        "dry_run": True,
        "truncated": False,
        "error": None,
    }
    payload.update(overrides)
    return Transcript.model_validate(payload)


def _write_transcript(
    results_root: Path,
    *,
    question_id: str,
    attempt: int,
    text: str,
    model_id: str = MODEL_ID,
    truncated: bool = False,
    error: str | None = None,
) -> Path:
    """Persist one transcript under ``results_root/wave-score/<model>/``."""
    transcript = _transcript(
        question_id=question_id,
        requested_model=model_id,
        response_model=model_id,
        attempt=attempt,
        text=text,
        truncated=truncated,
        error=error,
    )
    from maxq.runner import model_slug

    path = results_root / "wave-score" / model_slug(model_id) / f"{question_id}-a{attempt}.json"
    dump_transcript(transcript, path)
    return path


def test_scoring_import_does_not_raise() -> None:
    """Importing maxq.scoring must no longer raise NotImplementedError."""
    import maxq.scoring as scoring_mod

    assert callable(scoring_mod.main)


def test_extract_final_last_line_wins() -> None:
    """The last FINAL line is the parsed payload, even if earlier lines exist."""
    text = "scratch FINAL: 999 km\nworking...\nFINAL: 1.0 m/s\n"
    assert extract_final(text) == "1.0 m/s"


def test_extract_final_missing_and_empty() -> None:
    """Missing or empty FINAL payloads return None; matching is case-insensitive."""
    assert extract_final("no answer here") is None
    assert extract_final("FINAL:\n") is None
    assert extract_final("final: 2.5 km/s") == "2.5 km/s"


def test_parse_numeric_unit_conversion_and_si_prefix() -> None:
    """km/s and SI prefixes convert into the question unit."""
    km = parse_numeric("0.001 km/s", "m/s")
    prefix = parse_numeric("1000 mm/s", "m/s")
    scientific = parse_numeric("1.0e0 m/s", "m/s")
    bare = parse_numeric("1.0", "m/s")
    assert km is not None
    assert prefix is not None
    assert scientific is not None
    assert bare is not None
    truth = parse_numeric("1.0 m/s", "m/s")
    assert truth is not None
    assert within_rel_tol(km, truth, 0.01)
    assert within_rel_tol(prefix, truth, 0.01)
    assert within_rel_tol(scientific, truth, 0.01)
    assert within_rel_tol(bare, truth, 0.01)


def test_tolerance_boundary_inclusive() -> None:
    """Exactly rel_tol passes; just outside the band fails.

    Truth 100 and rel_tol 0.01 make the band edge 101 exact in IEEE floats,
    so the inclusive check is not an artifact of 1.01 - 1.0 rounding.
    """
    truth = parse_numeric("100.0 m/s", "m/s")
    at_bound = parse_numeric("101.0 m/s", "m/s")
    outside = parse_numeric("101.1 m/s", "m/s")
    assert truth is not None
    assert at_bound is not None
    assert outside is not None
    assert within_rel_tol(at_bound, truth, 0.01)
    assert not within_rel_tol(outside, truth, 0.01)


def test_numeric_unit_conversion_scores_correct() -> None:
    """0.001 km/s matches truth 1.0 m/s inside rel_tol."""
    question = _question()
    transcript = _transcript(text="working\nFINAL: 0.001 km/s")
    result = score_numeric(question, transcript)
    assert result.status == "correct"
    assert result.score == 1.0
    assert isinstance(result.parsed_answer, float)
    assert result.parsed_answer == pytest.approx(1.0)


def test_incompatible_units_score_zero_with_note() -> None:
    """A parsed quantity in the wrong dimension is 0 with an incompatible-units note."""
    question = _question()
    transcript = _transcript(text="FINAL: 1.0 kg")
    result = score_numeric(question, transcript)
    assert result.score == 0.0
    assert result.status == "unparseable"
    assert result.scorer_notes is not None
    assert "incompatible" in result.scorer_notes


def test_unparseable_without_final_is_zero() -> None:
    """A complete attempt with no FINAL line scores 0 and is marked unparseable."""
    question = _question()
    transcript = _transcript(text="I am not sure.")
    result = score_numeric(question, transcript)
    assert result.score == 0.0
    assert result.status == "unparseable"
    assert result.parsed_answer is None


def test_truncated_missing_final_is_not_scored_zero() -> None:
    """Token-cap truncation with no FINAL is truncated, not incorrect."""
    question = _question()
    transcript = _transcript(text="I was deriving the thrust equation", truncated=True)
    result = score_numeric(question, transcript)
    assert result.score is None
    assert result.status == "truncated"
    assert result.truncated is True


def test_non_numeric_final_is_unparseable() -> None:
    """FINAL with a non-quantity payload scores 0."""
    question = _question()
    transcript = _transcript(text="FINAL: potato")
    result = score_numeric(question, transcript)
    assert result.score == 0.0
    assert result.status == "unparseable"


def test_series_match_and_length_mismatch() -> None:
    """A matching JSON array passes; a different length fails."""
    question = _question(
        id="W0-SER-001",
        scoring="ground_truth_series",
        answer="[1.0, 2.0, 3.0]",
        domain="telemetry",
        difficulty="practitioner",
    )
    ok = score_series(
        question, _transcript(question_id="W0-SER-001", text="FINAL: [1.0, 2.0, 3.0]")
    )
    assert ok.status == "correct"
    assert ok.score == 1.0
    short = score_series(question, _transcript(question_id="W0-SER-001", text="FINAL: [1.0, 2.0]"))
    assert short.status == "incorrect"
    assert short.score == 0.0
    assert short.scorer_notes is not None
    assert "length" in short.scorer_notes


def test_series_element_outside_tolerance() -> None:
    """One element outside rel_tol fails the whole series."""
    question = _question(
        id="W0-SER-001",
        scoring="ground_truth_series",
        answer="[1.0, 2.0, 3.0]",
        domain="telemetry",
        difficulty="practitioner",
    )
    result = score_series(
        question,
        _transcript(question_id="W0-SER-001", text="FINAL: [1.0, 2.5, 3.0]"),
    )
    assert result.status == "incorrect"
    assert result.score == 0.0
    assert result.scorer_notes is not None
    assert "element 1" in result.scorer_notes


def test_exact_match_and_mismatch() -> None:
    """Exact scoring strips whitespace and compares case-sensitively."""
    question = _question(
        id="W0-EX-001",
        scoring="exact",
        answer="alpha",
        unit=None,
        rel_tol=None,
        domain="telemetry",
        difficulty="practitioner",
    )
    hit = score_exact(question, _transcript(question_id="W0-EX-001", text="FINAL: alpha"))
    miss = score_exact(question, _transcript(question_id="W0-EX-001", text="FINAL: Alpha"))
    assert hit.status == "correct"
    assert hit.score == 1.0
    assert miss.status == "incorrect"
    assert miss.score == 0.0


def test_missing_and_error_transcripts_are_unscored() -> None:
    """Missing files and error transcripts do not count as wrong answers."""
    question = _question()
    missing = score_attempt(question, None, model_id=MODEL_ID, attempt=1)
    errored = score_attempt(
        question,
        _transcript(error="TimeoutError: boom", text=""),
        model_id=MODEL_ID,
        attempt=1,
    )
    assert missing.status == "missing"
    assert missing.score is None
    assert errored.status == "error"
    assert errored.score is None


def test_rubric_pending_then_accept_and_override(tmp_path: Path) -> None:
    """Fake judge fills the queue; accept-llm confirms; an override is logged."""
    questions = load_questions(FIXTURE)
    rubric_q = next(item for item in questions if item.scoring == "rubric")
    for question in questions:
        for attempt in (1, 2):
            if question.scoring == "numeric_tolerance":
                text = "FINAL: 1.0 m/s" if attempt == 1 else "FINAL: 9.0 m/s"
            elif question.scoring == "exact":
                text = "FINAL: alpha"
            elif question.scoring == "ground_truth_series":
                text = "FINAL: [1.0, 2.0, 3.0]"
            else:
                text = "The governing equation is n-dot. Assumption: vacuum."
            _write_transcript(tmp_path, question_id=question.id, attempt=attempt, text=text)

    judge = FakeJudge([True, True])
    pending_report = score_wave(
        questions=questions,
        config=_config(attempts=2),
        models=[_spec()],
        results_root=tmp_path,
        wave_slug="score",
        judge=judge,
        accept_llm=False,
    )
    rubric_rows = [row for row in pending_report.attempts if row.question_id == rubric_q.id]
    assert all(row.status == "pending_rubric" for row in rubric_rows)
    assert all(row.score is None for row in rubric_rows)
    queue_payload = json.loads((tmp_path / "wave-score" / "rubric-queue.json").read_text())
    assert queue_payload[0]["status"] == "pending"
    assert queue_payload[0]["llm_scores"] == [True, True]
    assert judge.calls >= 1

    accepted = score_wave(
        questions=questions,
        config=_config(attempts=2),
        models=[_spec()],
        results_root=tmp_path,
        wave_slug="score",
        judge=None,
        accept_llm=True,
    )
    confirmed = [row for row in accepted.attempts if row.question_id == rubric_q.id]
    assert all(row.status == "correct" and row.score == 1.0 for row in confirmed)
    log_text = (tmp_path / "wave-score" / "overrides.jsonl").read_text(encoding="utf-8")
    assert '"action": "accept"' in log_text

    override = RubricOverrideSpec(
        question_id=rubric_q.id,
        model=MODEL_ID,
        attempt=1,
        criterion_index=1,
        to=False,
        reviewer="tester",
        reason="assumption never named",
    )
    overridden = score_wave(
        questions=questions,
        config=_config(attempts=2),
        models=[_spec()],
        results_root=tmp_path,
        wave_slug="score",
        judge=None,
        accept_llm=False,
        overrides=[override],
    )
    attempt_one = next(
        row for row in overridden.attempts if row.question_id == rubric_q.id and row.attempt == 1
    )
    assert attempt_one.status == "incorrect"
    assert attempt_one.score == 0.5
    assert attempt_one.criterion_scores == [True, False]
    log_lines = [
        json.loads(line)
        for line in (tmp_path / "wave-score" / "overrides.jsonl").read_text().splitlines()
        if line
    ]
    override_lines = [line for line in log_lines if line["action"] == "override"]
    assert len(override_lines) == 1
    assert override_lines[0]["from"] is True
    assert override_lines[0]["to"] is False
    assert override_lines[0]["reason"] == "assumption never named"


def test_accept_from_confirms_subset_and_keeps_all_models(tmp_path: Path) -> None:
    """--accept-from confirms listed rows only; other models stay in scored.json."""
    questions = load_questions(FIXTURE)
    rubric_q = next(item for item in questions if item.scoring == "rubric")
    other_id = "other-model"
    for model_id in (MODEL_ID, other_id):
        for question in questions:
            for attempt in (1, 2):
                if question.scoring == "numeric_tolerance":
                    text = "FINAL: 1.0 m/s"
                elif question.scoring == "exact":
                    text = "FINAL: alpha"
                elif question.scoring == "ground_truth_series":
                    text = "FINAL: [1.0, 2.0, 3.0]"
                else:
                    text = "The governing equation is n-dot. Assumption: vacuum."
                _write_transcript(
                    tmp_path,
                    question_id=question.id,
                    attempt=attempt,
                    text=text,
                    model_id=model_id,
                )
    models = [_spec(), _spec(other_id)]
    config = _config(attempts=2)
    config = config.model_copy(update={"models": models})
    score_wave(
        questions=questions,
        config=config,
        models=models,
        results_root=tmp_path,
        wave_slug="score",
        judge=FakeJudge([True, True]),
        accept_llm=False,
    )
    accepted = score_wave(
        questions=questions,
        config=config,
        models=models,
        results_root=tmp_path,
        wave_slug="score",
        judge=None,
        accept_from=[
            RubricAcceptSpec(question_id=rubric_q.id, model=MODEL_ID, attempt=1),
        ],
        reviewer="Darth",
    )
    assert [summary.model for summary in accepted.models] == [MODEL_ID, other_id]
    by_key = {
        (row.question_id, row.model, row.attempt): row
        for row in accepted.attempts
        if row.question_id == rubric_q.id
    }
    assert by_key[(rubric_q.id, MODEL_ID, 1)].status == "correct"
    assert by_key[(rubric_q.id, MODEL_ID, 2)].status == "pending_rubric"
    assert by_key[(rubric_q.id, other_id, 1)].status == "pending_rubric"
    log_lines = [
        json.loads(line)
        for line in (tmp_path / "wave-score" / "overrides.jsonl").read_text().splitlines()
        if line
    ]
    accepts = [line for line in log_lines if line["action"] == "accept"]
    assert len(accepts) == 1
    assert accepts[0]["reviewer"] == "Darth"
    assert accepts[0]["reason"] == "accept-from"


def test_accept_from_unknown_triple_errors(tmp_path: Path) -> None:
    """An accept-from row that is not in the queue is a hard error."""
    questions = load_questions(FIXTURE)
    rubric_q = next(item for item in questions if item.scoring == "rubric")
    for question in questions:
        text = "FINAL: 1.0 m/s"
        if question.scoring == "exact":
            text = "FINAL: alpha"
        elif question.scoring == "ground_truth_series":
            text = "FINAL: [1.0, 2.0, 3.0]"
        elif question.scoring == "rubric":
            text = "derivation"
        _write_transcript(tmp_path, question_id=question.id, attempt=1, text=text)
    score_wave(
        questions=questions,
        config=_config(attempts=1),
        models=[_spec()],
        results_root=tmp_path,
        wave_slug="score",
        judge=FakeJudge([True, True]),
    )
    with pytest.raises(WaveError, match="unknown rubric attempt"):
        score_wave(
            questions=questions,
            config=_config(attempts=1),
            models=[_spec()],
            results_root=tmp_path,
            wave_slug="score",
            accept_from=[
                RubricAcceptSpec(question_id=rubric_q.id, model="nope", attempt=1),
            ],
        )


def test_pass_at_1_false_best_of_n_true(tmp_path: Path) -> None:
    """Attempt 1 misses, attempt 2 hits: pass@1 is 0 and best-of-n is 1 for that tier."""
    question = _question()
    _write_transcript(tmp_path, question_id=question.id, attempt=1, text="FINAL: 9.0 m/s")
    _write_transcript(tmp_path, question_id=question.id, attempt=2, text="FINAL: 1.0 m/s")
    report = score_wave(
        questions=[question],
        config=_config(attempts=2),
        models=[_spec()],
        results_root=tmp_path,
        wave_slug="score",
    )
    summary = report.models[0]
    assert summary.pass_at_1 == 0.0
    assert summary.best_of_n == 1.0
    undergrad = summary.by_tier["undergrad"]
    assert undergrad.n_questions == 1
    assert undergrad.pass_at_1 == 0.0
    assert undergrad.best_of_n == 1.0
    table = format_score_table(report)
    assert "undergrad" in table
    assert "practitioner" in table
    assert "expert" in table
    assert table.count("\n") >= 4


def test_cli_table_includes_every_tier_not_a_headline(tmp_path: Path) -> None:
    """CLI stdout is a model×tier table covering all three difficulties."""
    questions = load_questions(FIXTURE)
    for question in questions:
        for attempt in (1, 2, 3):
            if question.scoring == "numeric_tolerance":
                text = "FINAL: 1.0 m/s"
            elif question.scoring == "exact":
                text = "FINAL: alpha"
            elif question.scoring == "ground_truth_series":
                text = "FINAL: [1.0, 2.0, 3.0]"
            else:
                text = "derivation covering both bullets"
            _write_transcript(
                tmp_path,
                question_id=question.id,
                attempt=attempt,
                text=text,
                model_id="x-ai/grok-4.6",
            )
    code = main(
        [
            "--questions",
            str(FIXTURE),
            "--wave",
            "score",
            "--results-root",
            str(tmp_path),
            "--config",
            str(CONFIG),
            "--model",
            "x-ai/grok-4.6",
            "--judge-stub",
            "--accept-llm",
        ]
    )
    assert code == 0
    scored = json.loads((tmp_path / "wave-score" / "scored.json").read_text(encoding="utf-8"))
    assert scored["wave"] == "wave-score"
    assert "undergrad" in scored["models"][0]["by_tier"]
    assert "practitioner" in scored["models"][0]["by_tier"]
    assert "expert" in scored["models"][0]["by_tier"]
    transcript_before = (
        tmp_path / "wave-score" / "x-ai--grok-4.6" / f"{questions[0].id}-a1.json"
    ).read_text(encoding="utf-8")
    assert "answer" not in json.loads(transcript_before)
    assert "rel_tol" not in json.loads(transcript_before)


def test_cli_prints_tier_table(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The CLI prints every tier name rather than a single headline score."""
    questions = load_questions(FIXTURE)
    for question in questions:
        text = "FINAL: 1.0 m/s"
        if question.scoring == "exact":
            text = "FINAL: alpha"
        elif question.scoring == "ground_truth_series":
            text = "FINAL: [1.0, 2.0, 3.0]"
        elif question.scoring == "rubric":
            text = "derivation"
        for attempt in (1, 2, 3):
            _write_transcript(
                tmp_path,
                question_id=question.id,
                attempt=attempt,
                text=text,
                model_id="x-ai/grok-4.6",
            )
    code = main(
        [
            "--questions",
            str(FIXTURE),
            "--wave",
            "score",
            "--results-root",
            str(tmp_path),
            "--config",
            str(CONFIG),
            "--model",
            "x-ai/grok-4.6",
            "--judge-stub",
        ]
    )
    assert code == 0
    captured = capsys.readouterr()
    stdout = captured.out
    assert "pass@1" in stdout
    assert "best-of-n" in stdout
    assert "undergrad" in stdout
    assert "practitioner" in stdout
    assert "expert" in stdout
    numeric_lines = [line for line in stdout.splitlines() if line.startswith("x-ai/grok-4.6")]
    assert len(numeric_lines) == 3


def test_provider_judge_parses_json_from_adapter() -> None:
    """ProviderRubricJudge reads criteria JSON from adapter visible text."""
    adapter = FakeAdapter('{"criteria": [true, false], "notes": "assumption missing"}')
    judge = ProviderRubricJudge(
        adapter=adapter,
        model_id="judge-model",
        config=_config(attempts=1),
        send_temperature=True,
    )
    question = load_questions(FIXTURE)
    rubric_q = next(item for item in question if item.scoring == "rubric")
    judgment = judge.score(
        prompt=rubric_q.prompt,
        transcript_text="states the equation only",
        rubric=list(rubric_q.rubric) if rubric_q.rubric is not None else [],
    )
    assert judgment.criterion_scores == [True, False]
    assert adapter.calls == 1


def test_parse_judge_json_fence_and_mismatch() -> None:
    """Judge JSON may be fenced; wrong-length criteria are rejected."""
    fenced = parse_judge_json('```json\n{"criteria": [true, true], "notes": "ok"}\n```', 2)
    assert fenced is not None
    assert fenced.criterion_scores == [True, True]
    assert parse_judge_json('{"criteria": [true], "notes": "short"}', 2) is None


def test_zero_truth_requires_zero_candidate() -> None:
    """A zero canonical answer only matches a converted zero candidate."""
    truth = parse_numeric("0.0 m/s", "m/s")
    zero = parse_numeric("0 km/s", "m/s")
    nonzero = parse_numeric("1.0 m/s", "m/s")
    assert truth is not None
    assert zero is not None
    assert nonzero is not None
    assert within_rel_tol(zero, truth, 0.01)
    assert not within_rel_tol(nonzero, truth, 0.01)


def test_stub_judge_marks_all_false() -> None:
    """The CLI stub judge never awards rubric bullets."""
    rubric = ["States the governing equation", "Names the assumption"]
    judgment = StubRubricJudge().score(
        prompt="x",
        transcript_text="y",
        rubric=rubric,
    )
    assert judgment.criterion_scores == [False, False]


def test_load_run_config_still_valid() -> None:
    """Scoring tests do not require mutating the committed run config."""
    config = load_run_config(CONFIG)
    assert config.attempts == 3


def test_rubric_queue_records_judge_identity(tmp_path: Path) -> None:
    """The published queue says which model produced the first-pass scores."""
    from maxq.runner import default_config_path, load_run_config
    from maxq.scoring import StubRubricJudge, score_wave
    from maxq.wave import load_questions

    config = load_run_config(default_config_path())
    questions = [q for q in load_questions(FIXTURE) if q.scoring == "rubric"]
    assert questions, "scoring fixture needs a rubric question"
    # dry-run a wave so transcripts exist
    from maxq.runner import main as runner_main

    assert (
        runner_main(
            [
                "--questions",
                str(FIXTURE),
                "--wave",
                "judgeid",
                "--results-root",
                str(tmp_path),
                "--dry-run",
            ]
        )
        == 0
    )
    score_wave(
        questions=questions,
        config=config,
        models=[spec for spec in config.models if spec.enabled],
        results_root=tmp_path,
        wave_slug="judgeid",
        judge=StubRubricJudge(),
    )
    queue = json.loads((tmp_path / "wave-judgeid" / "rubric-queue.json").read_text())
    assert queue, "queue should have rubric rows"
    assert all(item["judge_model"] == "stub" for item in queue)
    keys = list(queue[0])
    assert keys.index("judge_model") == keys.index("status") - 1
