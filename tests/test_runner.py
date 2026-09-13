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
    providers = {spec.provider for spec in enabled}
    assert providers == {"xai", "anthropic", "openai", "google"}
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
    ledger = CostLedger.model_validate(json.loads((wave_dir / "cost.json").read_text(encoding="utf-8")))
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
    ledger2 = CostLedger.model_validate(json.loads((wave_dir / "cost.json").read_text(encoding="utf-8")))
    assert ledger2.totals.cost_usd == summed
    assert ledger2.totals.calls == first_calls
    assert ledger2.totals.skipped == first_calls


def test_resume_retries_error_transcripts(tmp_path: Path) -> None:
    """A transcript with error set is overwritten on the next dry-run."""
    assert _run_dry(tmp_path, extra=["--model", "grok-4.6"]) == 0
    path = tmp_path / "wave-smoke" / "grok-4.6" / "W0-TEST-001-a1.json"
    transcript = Transcript.model_validate(json.loads(path.read_text(encoding="utf-8")))
    transcript.error = "injected failure"
    path.write_text(transcript.model_dump_json(indent=2) + "\n", encoding="utf-8")
    assert load_complete_transcript(path) is None
    assert _run_dry(tmp_path, extra=["--model", "grok-4.6"]) == 0
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
            "grok-4.6",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    files = list((tmp_path / "wave-smoke" / "grok-4.6").glob("*-a*.json"))
    assert len(files) == 9


def test_no_generation_branch_in_harness_code() -> None:
    """A later Grok generation is a config row, not an adapter ``if``."""
    for rel in ("maxq/providers.py", "maxq/runner.py"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert "4.7" not in text


def test_sanitize_redacts_credentials_not_usage() -> None:
    """``api_key`` is redacted; ``input_tokens`` is left intact."""
    cleaned = sanitize({"api_key": "sk-secret-value", "input_tokens": 9, "nested": {"password": "x"}})
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
    for name in ("XAI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"):
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
            "grok-4.6",
        ]
    )
    assert code == 1
    assert list(tmp_path.glob("**/*.json")) == []


def test_verify_models_dry_run_no_network() -> None:
    """``--verify-models --dry-run`` pings the dry-run adapter and exits 0."""
    code = main(["--verify-models", "--dry-run", "--config", str(CONFIG), "--model", "grok-4.6"])
    assert code == 0


def test_retry_on_429_then_success() -> None:
    """Retryable HTTP 429 is retried; HTTP 400 is not."""
    flaky = FlakyAdapter(fail_times=2, status_code=429)
    result = call_with_retry(
        flaky,
        retry_max=5,
        retry_base_s=0.0,
        sleep=False,
        model_id="grok-4.6",
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
            model_id="grok-4.6",
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
        model = "grok-4.6"
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
        model_id="grok-4.6",
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
    """Committed config lists the four providers plus the small-model floor."""
    config = load_run_config(CONFIG)
    ids = [spec.id for spec in config.models if spec.enabled]
    assert ids == [
        "grok-4.6",
        "claude-fable-5-1",
        "gpt-6-astra",
        "gemini-3.1-pro",
        "gemini-3.8-flash",
    ]
    assert config.temperature == 0.0
    assert config.attempts == 3
    assert config.prompt_template_id == "maxq-closed-book-v1"


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
