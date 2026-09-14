"""Tests for the identical-treatment run harness (issue #2)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest
from google.genai import types as genai_types

from maxq.providers import (
    AnthropicAdapter,
    GoogleAdapter,
    OpenAIAdapter,
    XAIAdapter,
    call_with_retry,
    sanitize,
)
from maxq.runner import (
    default_config_path,
    estimate_cost_usd,
    load_complete_transcript,
    load_run_config,
    main,
    render_prompt,
    run_config_sha256,
)
from maxq.schema import AdapterResult, CostLedger, TokenUsage, Transcript
from maxq.wave import load_questions

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).parent / "fixtures" / "sample_wave.json"
CONFIG = ROOT / "config" / "run.json"

FORBIDDEN_KEYS = {"answer", "provenance", "rel_tol", "rubric", "api_key"}
LEAK_SUBSTRINGS = ("1.234567", "Synthetic fixture for unit tests")


class FakeHTTPError(Exception):
    """SDK-shaped HTTP error with ``status_code`` for retry tests."""

    def __init__(self, status_code: int, message: str = "http error") -> None:
        self.status_code = status_code
        super().__init__(message)


class FlakyAdapter:
    """Raises retryable errors for the first ``fail_times`` calls, then succeeds."""

    def __init__(self, fail_times: int, status_code: int) -> None:
        self.fail_times = fail_times
        self.status_code = status_code
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
        """Fail ``fail_times`` times, then return a tiny success result."""
        del system, user, temperature, send_temperature, max_output_tokens, timeout_s
        self.calls += 1
        if self.calls <= self.fail_times:
            raise FakeHTTPError(self.status_code)
        return AdapterResult(
            text="ok",
            usage=TokenUsage(input_tokens=1, output_tokens=1),
            raw_request={"model": model_id},
            raw_response={"text": "ok"},
            request_temperature=None,
            response_model=model_id,
            sdk_version="test",
        )


def _json_keys(value: object) -> set[str]:
    """Collect object keys from nested JSON."""
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            keys.add(str(key))
            keys |= _json_keys(item)
    elif isinstance(value, list):
        for item in value:
            keys |= _json_keys(item)
    return keys


def _run_dry(
    tmp_path: Path,
    extra: list[str] | None = None,
) -> int:
    """Invoke the runner CLI in-process with the smoke fixture."""
    argv = [
        "--questions",
        str(FIXTURE),
        "--wave",
        "smoke",
        "--dry-run",
        "--results-root",
        str(tmp_path),
        "--config",
        str(CONFIG),
    ]
    if extra:
        argv.extend(extra)
    return main(argv)


def test_runner_import_does_not_raise() -> None:
    """Importing maxq.runner must no longer raise NotImplementedError."""
    import maxq.runner as runner_mod

    assert callable(runner_mod.main)


def test_dry_run_smoke_writes_well_formed_transcripts(tmp_path: Path) -> None:
    """3 questions × enabled models × 3 attempts, all valid Transcript files."""
    assert _run_dry(tmp_path) == 0
    config = load_run_config(CONFIG)
    questions = load_questions(FIXTURE)
    enabled = [spec for spec in config.models if spec.enabled]
    assert len(enabled) > 0
    wave_dir = tmp_path / "wave-smoke"
    files = list(wave_dir.glob("*/*-a*.json"))
    assert len(files) == len(questions) * len(enabled) * config.attempts
    by_question: dict[str, list[list[dict[str, str]]]] = {q.id: [] for q in questions}
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        transcript = Transcript.model_validate(payload)
        assert transcript.error is None
        assert "FINAL:" in transcript.text
        assert transcript.dry_run is True
        assert transcript.attempt >= 1
        keys = _json_keys(payload)
        assert keys.isdisjoint(FORBIDDEN_KEYS)
        text = path.read_text(encoding="utf-8")
        for leak in LEAK_SUBSTRINGS:
            assert leak not in text
        rendered = [item.model_dump() for item in transcript.rendered_messages]
        by_question[transcript.question_id].append(rendered)
    for question_id, rendered_list in by_question.items():
        first = rendered_list[0]
        for rendered in rendered_list[1:]:
            assert rendered == first, question_id


def test_dry_run_cost_matches_transcripts_and_resume_does_not_duplicate(
    tmp_path: Path,
) -> None:
    """Cost ledger equals summed transcript costs; a second run skips and does not double."""
    assert _run_dry(tmp_path) == 0
    wave_dir = tmp_path / "wave-smoke"
    ledger = CostLedger.model_validate(
        json.loads((wave_dir / "cost.json").read_text(encoding="utf-8"))
    )
    transcripts = [
        Transcript.model_validate(json.loads(path.read_text(encoding="utf-8")))
        for path in wave_dir.glob("*/*-a*.json")
    ]
    summed = round(sum(item.cost_usd for item in transcripts), 8)
    assert ledger.totals.cost_usd == summed
    config = load_run_config(CONFIG)
    by_id = {spec.id: spec for spec in config.models}
    for item in transcripts:
        spec = by_id[item.requested_model]
        assert item.cost_usd == estimate_cost_usd(item.usage, spec.pricing_usd_per_mtok)
    first_calls = ledger.totals.calls
    assert _run_dry(tmp_path) == 0
    ledger2 = CostLedger.model_validate(
        json.loads((wave_dir / "cost.json").read_text(encoding="utf-8"))
    )
    assert ledger2.totals.cost_usd == summed
    assert ledger2.totals.calls == first_calls
    assert ledger2.totals.skipped == first_calls


def test_resume_retries_error_transcripts(tmp_path: Path) -> None:
    """A transcript with error set is overwritten on the next dry-run."""
    assert _run_dry(tmp_path, extra=["--model", "x-ai/grok-4.6"]) == 0
    path = tmp_path / "wave-smoke" / "x-ai--grok-4.6" / "W0-TEST-001-a1.json"
    transcript = Transcript.model_validate(json.loads(path.read_text(encoding="utf-8")))
    transcript.error = "injected failure"
    path.write_text(transcript.model_dump_json(indent=2) + "\n", encoding="utf-8")
    assert load_complete_transcript(path) is None
    assert _run_dry(tmp_path, extra=["--model", "x-ai/grok-4.6"]) == 0
    again = Transcript.model_validate(json.loads(path.read_text(encoding="utf-8")))
    assert again.error is None
    assert again.text.startswith("DRY-RUN")


def test_module_cli_dry_run_subprocess(tmp_path: Path) -> None:
    """``python -m maxq.runner`` dry-run of the 3-question smoke set exits 0."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "maxq.runner",
            "--questions",
            str(FIXTURE),
            "--wave",
            "smoke",
            "--dry-run",
            "--results-root",
            str(tmp_path),
            "--config",
            str(CONFIG),
            "--model",
            "x-ai/grok-4.6",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    files = list((tmp_path / "wave-smoke" / "x-ai--grok-4.6").glob("*-a*.json"))
    assert len(files) == 9


def test_no_generation_branch_in_harness_code() -> None:
    """A later Grok generation is a config row, not an adapter ``if``."""
    for rel in ("maxq/providers.py", "maxq/runner.py"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert "4.7" not in text


def test_sanitize_redacts_credentials_not_usage() -> None:
    """``api_key`` is redacted; ``input_tokens`` is left intact."""
    cleaned = sanitize(
        {"api_key": "sk-secret-value", "input_tokens": 9, "nested": {"password": "x"}}
    )
    assert isinstance(cleaned, dict)
    assert cleaned["api_key"] == "[redacted]"
    assert cleaned["input_tokens"] == 9
    nested = cleaned["nested"]
    assert isinstance(nested, dict)
    assert nested["password"] == "[redacted]"


def test_render_prompt_does_not_inject_answer_keys() -> None:
    """Canonical user text is the public prompt plus suffix, never the answer field."""
    config = load_run_config(CONFIG)
    question = load_questions(FIXTURE)[0]
    messages = render_prompt(question, config)
    blob = "\n".join(item.content for item in messages)
    assert question.prompt in blob
    assert "1.234567" not in blob
    assert question.provenance not in blob
    assert config.system_prompt == messages[0].content


def test_live_run_fail_fast_without_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-dry run exits 1 before any call when API keys are missing."""
    for name in (
        "XAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    code = main(
        [
            "--questions",
            str(FIXTURE),
            "--wave",
            "smoke",
            "--results-root",
            str(tmp_path),
            "--config",
            str(CONFIG),
            "--model",
            "x-ai/grok-4.6",
        ]
    )
    assert code == 1
    assert list(tmp_path.glob("**/*.json")) == []


def test_verify_models_dry_run_no_network() -> None:
    """``--verify-models --dry-run`` pings the dry-run adapter and exits 0."""
    code = main(
        ["--verify-models", "--dry-run", "--config", str(CONFIG), "--model", "x-ai/grok-4.6"]
    )
    assert code == 0


def test_retry_on_429_then_success() -> None:
    """Retryable HTTP 429 is retried; HTTP 400 is not."""
    flaky = FlakyAdapter(fail_times=2, status_code=429)
    result = call_with_retry(
        flaky,
        retry_max=5,
        retry_base_s=0.0,
        sleep=False,
        model_id="x-ai/grok-4.6",
        system="s",
        user="u",
        temperature=0.0,
        send_temperature=False,
        max_output_tokens=16,
        timeout_s=1.0,
    )
    assert result.text == "ok"
    assert flaky.calls == 3

    hard = FlakyAdapter(fail_times=5, status_code=400)
    with pytest.raises(FakeHTTPError):
        call_with_retry(
            hard,
            retry_max=5,
            retry_base_s=0.0,
            sleep=False,
            model_id="x-ai/grok-4.6",
            system="s",
            user="u",
            temperature=0.0,
            send_temperature=False,
            max_output_tokens=16,
            timeout_s=1.0,
        )
    assert hard.calls == 1


def test_run_config_sha256_matches_file_bytes() -> None:
    """Config digest is of the on-disk bytes, not a re-serialized object."""
    assert run_config_sha256(CONFIG) == sha256(CONFIG.read_bytes()).hexdigest()
    assert default_config_path() == CONFIG


def _patch_openai(
    monkeypatch: pytest.MonkeyPatch,
    factory: Callable[..., object],
) -> None:
    """Replace ``maxq.providers.OpenAI`` with ``factory``."""
    monkeypatch.setattr("maxq.providers.OpenAI", factory)


def test_openai_adapter_store_false_omits_tools_and_temperature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenAI Responses calls use store=False, no tools, and omit temperature when flagged."""
    captured: dict[str, object] = {}

    class FakeResponse:
        output_text = "visible\nFINAL: 1"
        model = "gpt-6-astra"
        usage = SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            input_tokens_details=SimpleNamespace(cached_tokens=0),
        )

        @property
        def output(self) -> list[object]:
            return []

        def model_dump(self, mode: str = "python") -> dict[str, object]:
            del mode
            return {"output_text": self.output_text, "model": self.model}

    class FakeResponses:
        def create(self, **kwargs: object) -> FakeResponse:
            captured.update(kwargs)
            return FakeResponse()

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            del kwargs
            self.responses = FakeResponses()

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    _patch_openai(monkeypatch, FakeClient)
    result = OpenAIAdapter().complete(
        model_id="gpt-6-astra",
        system="SYS",
        user="USR",
        temperature=0.0,
        send_temperature=False,
        max_output_tokens=128,
        timeout_s=5.0,
    )
    assert captured["store"] is False
    assert "tools" not in captured
    assert "temperature" not in captured
    assert captured["instructions"] == "SYS"
    assert captured["input"] == "USR"
    assert result.text == "visible\nFINAL: 1"
    assert "sk-test-openai" not in json.dumps(result.raw_request)


def test_xai_adapter_no_tools_canonical_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    """xAI chat completions send system+user and do not pass tools or n."""
    captured: dict[str, object] = {}
    init_kwargs: dict[str, object] = {}

    class FakeResponse:
        model = "x-ai/grok-4.6"
        usage = SimpleNamespace(
            prompt_tokens=11,
            completion_tokens=4,
            prompt_tokens_details=SimpleNamespace(cached_tokens=2),
        )

        @property
        def choices(self) -> list[SimpleNamespace]:
            return [SimpleNamespace(message=SimpleNamespace(content="ok\nFINAL: 1"))]

        def model_dump(self, mode: str = "python") -> dict[str, object]:
            del mode
            return {"model": self.model}

    class FakeCompletions:
        def create(self, **kwargs: object) -> FakeResponse:
            captured.update(kwargs)
            return FakeResponse()

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            init_kwargs.update(kwargs)
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setenv("XAI_API_KEY", "sk-test-xai")
    _patch_openai(monkeypatch, FakeClient)
    result = XAIAdapter().complete(
        model_id="x-ai/grok-4.6",
        system="SYS",
        user="USR",
        temperature=0.0,
        send_temperature=False,
        max_output_tokens=64,
        timeout_s=5.0,
    )
    assert str(init_kwargs.get("base_url")).rstrip("/") == "https://api.x.ai/v1"
    assert "tools" not in captured
    assert "n" not in captured
    assert "temperature" not in captured
    messages = captured["messages"]
    assert messages == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USR"},
    ]
    assert result.usage.cached_input_tokens == 2
    assert "sk-test-xai" not in json.dumps(result.raw_request)


def test_anthropic_adapter_skips_thinking_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Anthropic text extraction skips thinking blocks; system is a top-level arg."""
    captured: dict[str, object] = {}

    class FakeMessage:
        model = "claude-fable-5-1"
        usage = SimpleNamespace(input_tokens=8, output_tokens=6, cache_read_input_tokens=1)

        @property
        def content(self) -> list[SimpleNamespace]:
            return [
                SimpleNamespace(type="thinking", thinking="hidden"),
                SimpleNamespace(type="text", text="shown\nFINAL: 1"),
            ]

        def model_dump(self, mode: str = "python") -> dict[str, object]:
            del mode
            return {"model": self.model, "content": [{"type": "text", "text": "shown\nFINAL: 1"}]}

    class FakeMessages:
        def create(self, **kwargs: object) -> FakeMessage:
            captured.update(kwargs)
            return FakeMessage()

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            del kwargs
            self.messages = FakeMessages()

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-anthropic")
    monkeypatch.setattr("maxq.providers.anthropic.Anthropic", FakeClient)
    result = AnthropicAdapter().complete(
        model_id="claude-fable-5-1",
        system="SYS",
        user="USR",
        temperature=0.0,
        send_temperature=False,
        max_output_tokens=64,
        timeout_s=5.0,
    )
    assert captured["system"] == "SYS"
    assert captured["messages"] == [{"role": "user", "content": "USR"}]
    assert "temperature" not in captured
    assert result.text == "shown\nFINAL: 1"
    assert "hidden" not in result.text
    assert result.usage.cached_input_tokens == 1
    assert "sk-test-anthropic" not in json.dumps(result.raw_request)


def test_google_adapter_block_none_and_canonical_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """Google adapter sets BLOCK_NONE safety and does not append prompt text."""
    captured: dict[str, object] = {}

    class FakeResponse:
        text = "ok\nFINAL: 1"
        model_version = "gemini-3.1-pro"
        usage_metadata = SimpleNamespace(
            prompt_token_count=9,
            candidates_token_count=3,
            thoughts_token_count=2,
            cached_content_token_count=0,
        )

        def model_dump(self, mode: str = "python") -> dict[str, object]:
            del mode
            return {"text": self.text, "model_version": self.model_version}

    class FakeModels:
        def generate_content(self, **kwargs: object) -> FakeResponse:
            captured.update(kwargs)
            return FakeResponse()

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            del kwargs
            self.models = FakeModels()

    monkeypatch.setenv("GOOGLE_API_KEY", "sk-test-google")
    monkeypatch.setattr("maxq.providers.genai.Client", FakeClient)
    result = GoogleAdapter().complete(
        model_id="gemini-3.1-pro",
        system="SYS",
        user="USR",
        temperature=0.0,
        send_temperature=True,
        max_output_tokens=64,
        timeout_s=5.0,
    )
    assert captured["contents"] == "USR"
    config = captured["config"]
    assert isinstance(config, genai_types.GenerateContentConfig)
    assert config.system_instruction == "SYS"
    assert config.temperature == 0.0
    settings = config.safety_settings
    assert settings is not None
    assert all(item.threshold == genai_types.HarmBlockThreshold.BLOCK_NONE for item in settings)
    assert result.usage.output_tokens == 5
    assert "sk-test-google" not in json.dumps(result.raw_request)


def test_config_pins_expected_models() -> None:
    """Committed config routes every row through OpenRouter; all five enabled."""
    config = load_run_config(CONFIG)
    ids = [spec.id for spec in config.models]
    assert ids == [
        "x-ai/grok-4.6",
        "anthropic/claude-fable-5-1",
        "openai/gpt-6-astra",
        "google/gemini-3.1-pro",
        "google/gemini-3.8-flash",
    ]
    assert all(spec.provider == "openrouter" for spec in config.models)
    assert all(spec.enabled for spec in config.models)
    assert config.temperature == 0.0
    assert config.attempts == 3
    assert config.prompt_template_id == "maxq-closed-book-v1"


def test_openrouter_adapter_pins_routing_and_records_served_by(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenRouter calls disable fallbacks, pin order, and surface served_by."""
    from maxq.providers import OpenRouterAdapter

    captured: dict[str, object] = {}

    class FakeCompletions:
        def create(self, **kwargs: object) -> object:
            captured.update(kwargs)
            return SimpleNamespace(
                model="openai/gpt-6-astra",
                provider="OpenAI",
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(content="ok\nFINAL: 1 m/s"),
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=10,
                    completion_tokens=5,
                    prompt_tokens_details=None,
                ),
                model_dump=lambda mode="python": {"model": "openai/gpt-6-astra"},
            )

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            captured["base_url"] = kwargs.get("base_url")
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-openrouter")
    monkeypatch.setattr("maxq.providers.OpenAI", FakeClient)
    result = OpenRouterAdapter(providers_order=["OpenAI"]).complete(
        model_id="openai/gpt-6-astra",
        system="SYS",
        user="USR",
        temperature=0.0,
        send_temperature=False,
        max_output_tokens=64,
        timeout_s=5.0,
    )
    assert captured["base_url"] == "https://openrouter.ai/api/v1"
    extra = captured["extra_body"]
    assert extra == {"provider": {"allow_fallbacks": False, "order": ["OpenAI"]}}
    assert "temperature" not in captured
    assert result.served_by == "OpenAI"
    assert result.response_model == "openai/gpt-6-astra"
    assert result.truncated is False
    assert "sk-test-openrouter" not in json.dumps(result.raw_request)


def test_transcript_served_by_key_position(tmp_path: Path) -> None:
    """served_by is persisted right after response_model in transcript JSON."""
    from maxq.runner import _success_transcript, dump_transcript
    from maxq.schema import RenderedMessage

    config = load_run_config(CONFIG)
    spec = config.models[0]
    questions = load_questions(FIXTURE)
    result = AdapterResult(
        text="FINAL: 1 m/s",
        usage=TokenUsage(input_tokens=1, output_tokens=1),
        raw_request={},
        raw_response={},
        request_temperature=None,
        response_model=spec.id,
        sdk_version="test",
        served_by="xAI",
    )
    transcript = _success_transcript(
        question=questions[0],
        spec=spec,
        attempt=1,
        wave="wave-smoke",
        config=config,
        config_sha="0" * 64,
        messages=[
            RenderedMessage(role="system", content="s"),
            RenderedMessage(role="user", content="u"),
        ],
        max_output_tokens=64,
        dry_run=True,
        started_at="2026-09-14T00:00:00Z",
        result=result,
    )
    path = tmp_path / "t.json"
    dump_transcript(transcript, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["served_by"] == "xAI"
    keys = list(payload)
    assert keys.index("served_by") == keys.index("response_model") + 1


def test_empty_questions_rejected(tmp_path: Path) -> None:
    """An empty question array exits 1."""
    path = tmp_path / "empty.json"
    path.write_text("[]\n", encoding="utf-8")
    code = main(
        [
            "--questions",
            str(path),
            "--wave",
            "smoke",
            "--dry-run",
            "--results-root",
            str(tmp_path / "out"),
            "--config",
            str(CONFIG),
        ]
    )
    assert code == 1


def test_truncation_detectors() -> None:
    """Each provider's stop-at-cap shape maps to truncated=True."""
    from maxq.providers import (
        _anthropic_truncated,
        _chat_truncated,
        _google_truncated,
        _responses_truncated,
    )

    chat_hit = SimpleNamespace(choices=[SimpleNamespace(finish_reason="length")])
    chat_ok = SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop")])
    assert _chat_truncated(chat_hit) is True
    assert _chat_truncated(chat_ok) is False

    resp_hit = SimpleNamespace(
        status="incomplete",
        incomplete_details=SimpleNamespace(reason="max_output_tokens"),
    )
    resp_ok = SimpleNamespace(status="completed", incomplete_details=None)
    assert _responses_truncated(resp_hit) is True
    assert _responses_truncated(resp_ok) is False

    assert _anthropic_truncated(SimpleNamespace(stop_reason="max_tokens")) is True
    assert _anthropic_truncated(SimpleNamespace(stop_reason="end_turn")) is False

    google_hit = SimpleNamespace(
        candidates=[SimpleNamespace(finish_reason=genai_types.FinishReason.MAX_TOKENS)]
    )
    google_ok = SimpleNamespace(
        candidates=[SimpleNamespace(finish_reason=genai_types.FinishReason.STOP)]
    )
    assert _google_truncated(google_hit) is True
    assert _google_truncated(google_ok) is False
    assert _google_truncated(SimpleNamespace(candidates=[])) is False


def test_transcript_records_truncated_flag(tmp_path: Path) -> None:
    """A truncated adapter result lands as truncated=true in the transcript JSON."""
    from maxq.runner import _success_transcript, dump_transcript
    from maxq.schema import RenderedMessage

    config = load_run_config(CONFIG)
    spec = next(item for item in config.models if item.id.startswith("google/"))
    questions = load_questions(FIXTURE)
    result = AdapterResult(
        text="",
        usage=TokenUsage(input_tokens=10, output_tokens=spec.max_output_tokens or 1),
        raw_request={},
        raw_response={},
        request_temperature=0.0,
        response_model=spec.id,
        sdk_version="test",
        truncated=True,
    )
    transcript = _success_transcript(
        question=questions[0],
        spec=spec,
        attempt=1,
        wave="wave-smoke",
        config=config,
        config_sha="0" * 64,
        messages=[
            RenderedMessage(role="system", content="s"),
            RenderedMessage(role="user", content="u"),
        ],
        max_output_tokens=32768,
        dry_run=True,
        started_at="2026-09-13T00:00:00Z",
        result=result,
    )
    path = tmp_path / "t.json"
    dump_transcript(transcript, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["truncated"] is True
    assert payload["max_output_tokens"] == 32768
    keys = list(payload)
    assert keys.index("truncated") == keys.index("error") - 1


def test_google_models_have_token_budget_override() -> None:
    """Config gives Google rows headroom for thinking tokens; others use run level."""
    from maxq.runner import effective_max_output_tokens

    config = load_run_config(CONFIG)
    for spec in config.models:
        effective = effective_max_output_tokens(spec, config)
        if spec.id.startswith("google/"):
            assert spec.max_output_tokens is not None
            assert effective == spec.max_output_tokens
            assert effective > config.max_output_tokens
        else:
            assert effective == config.max_output_tokens


def test_verify_ping_budget_not_one_token() -> None:
    """Verify pings use a reasoning-safe budget, not max_output_tokens=1."""
    from maxq.runner import VERIFY_MAX_TOKENS

    assert VERIFY_MAX_TOKENS >= 256


def test_wave_results_are_gitignored_and_publishable_only_by_script(tmp_path: Path) -> None:
    """results/wave-*/ transcripts never enter git without the publish script."""
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "results/wave-*/" in ignore
    check = subprocess.run(
        ["git", "check-ignore", "results/wave-1/some-model/W1-PROP-001-a1.json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert check.returncode == 0, "wave transcript path is not gitignored"
    tracked = subprocess.run(
        ["git", "ls-files", "results"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert tracked.stdout.strip() == "results/README.md"


def test_publish_wave_refuses_incomplete_wave(tmp_path: Path) -> None:
    """The publish guard exits non-zero while any enabled model lacks transcripts."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "publish_wave", ROOT / "scripts" / "publish_wave.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    code = module.main(
        [
            "--wave",
            "77",
            "--questions",
            str(FIXTURE),
            "--results-root",
            str(tmp_path),
        ]
    )
    assert code == 2
