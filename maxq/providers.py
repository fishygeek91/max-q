"""Provider adapters: identical canonical text, provider-specific wire format.

Live clients are constructed on first ``complete`` call (importing this module
does not require API keys). Dry-run never constructs an SDK client.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from importlib.metadata import PackageNotFoundError, version
from typing import Protocol

import anthropic
from google import genai
from google.genai import types as genai_types
from openai import OpenAI

from maxq.schema import AdapterResult, ProviderName, TokenUsage

XAI_BASE_URL = "https://api.x.ai/v1"
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 529})
PROVIDER_ENV: dict[ProviderName, tuple[str, ...]] = {
    "xai": ("XAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "google": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
}
_REDACT_EXACT = frozenset(
    {
        "api_key",
        "access_token",
        "authorization",
        "secret",
        "password",
        "token",
        "x_api_key",
        "apikey",
    }
)
_GOOGLE_HARM_CATEGORIES: tuple[genai_types.HarmCategory, ...] = (
    genai_types.HarmCategory.HARM_CATEGORY_HARASSMENT,
    genai_types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
    genai_types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
    genai_types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
    genai_types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
    genai_types.HarmCategory.HARM_CATEGORY_JAILBREAK,
)


class ProviderAdapter(Protocol):
    """One completion call. Implementations must not append prompt text."""

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
        """Send the canonical system/user pair and return a normalized result."""


class MissingAPIKeyError(RuntimeError):
    """Raised when a live adapter is used without its provider environment key."""


def package_version(distribution: str) -> str:
    """Return the installed distribution version, or ``unknown`` if missing."""
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "unknown"


def env_key(provider: ProviderName) -> str | None:
    """Return the first set environment variable for ``provider``, else None."""
    for name in PROVIDER_ENV[provider]:
        value = os.environ.get(name)
        if value is not None and value.strip():
            return value
    return None


def missing_key_errors(provider: ProviderName, model_id: str) -> str | None:
    """Describe a missing key for ``model_id``, or None if a key is present."""
    if env_key(provider) is not None:
        return None
    names = " or ".join(PROVIDER_ENV[provider])
    return f"{model_id}: missing {names}"


def require_key(provider: ProviderName) -> str:
    """Return the provider API key or raise ``MissingAPIKeyError``."""
    value = env_key(provider)
    if value is None:
        names = " or ".join(PROVIDER_ENV[provider])
        raise MissingAPIKeyError(f"missing {names}")
    return value


def jsonable(value: object) -> object:
    """Convert an SDK object into JSON-compatible Python (dict/list/scalars)."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            dumped: object = dump(mode="json")
        except TypeError:
            dumped = dump()
        return jsonable(dumped)
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [jsonable(item) for item in value]
    return {"_repr": repr(value)}


def jsonable_object(value: object) -> dict[str, object]:
    """Like ``jsonable`` but always a dict (wraps non-dicts under ``value``)."""
    converted = jsonable(value)
    if isinstance(converted, dict):
        return {str(key): item for key, item in converted.items()}
    return {"value": converted}


def _is_secret_key(name: str) -> bool:
    """True for credential-like keys; false for usage fields such as input_tokens."""
    lower = name.lower().replace("-", "_")
    if lower in _REDACT_EXACT:
        return True
    return lower.endswith(("_api_key", "_secret", "_password"))


def sanitize(value: object) -> object:
    """Redact credential keys recursively. Does not touch ``*_tokens`` usage fields."""
    if isinstance(value, Mapping):
        out: dict[str, object] = {}
        for key, item in value.items():
            name = str(key)
            if _is_secret_key(name):
                out[name] = "[redacted]"
            else:
                out[name] = sanitize(item)
        return out
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [sanitize(item) for item in value]
    return value


def sanitize_object(value: object) -> dict[str, object]:
    """Sanitize and coerce to ``dict[str, object]`` for transcript fields."""
    cleaned = sanitize(jsonable_object(value))
    if isinstance(cleaned, dict):
        return {str(key): item for key, item in cleaned.items()}
    return {"value": cleaned}


def http_status(exc: BaseException) -> int | None:
    """Extract an HTTP status code from an SDK exception, if present."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    code = getattr(exc, "code", None)
    if isinstance(code, int):
        return code
    return None


def is_retryable(exc: BaseException) -> bool:
    """True when ``exc`` is a retryable HTTP status (429/5xx/529)."""
    status = http_status(exc)
    return status is not None and status in RETRYABLE_STATUS


def call_with_retry(
    adapter: ProviderAdapter,
    *,
    retry_max: int,
    retry_base_s: float,
    sleep: bool,
    model_id: str,
    system: str,
    user: str,
    temperature: float,
    send_temperature: bool,
    max_output_tokens: int,
    timeout_s: float,
) -> AdapterResult:
    """Call ``adapter.complete``, retrying retryable HTTP statuses up to ``retry_max``."""
    last_error: BaseException | None = None
    for attempt_index in range(retry_max):
        try:
            return adapter.complete(
                model_id=model_id,
                system=system,
                user=user,
                temperature=temperature,
                send_temperature=send_temperature,
                max_output_tokens=max_output_tokens,
                timeout_s=timeout_s,
            )
        except Exception as exc:
            last_error = exc
            if not is_retryable(exc) or attempt_index + 1 >= retry_max:
                raise
            if sleep:
                delay = retry_base_s * (2**attempt_index)
                time.sleep(delay)
    if last_error is None:
        raise RuntimeError("retry loop exited without a result or error")
    raise last_error


def adapter_for(provider: ProviderName, *, dry_run: bool) -> ProviderAdapter:
    """Return a dry-run adapter or the live adapter for ``provider``."""
    if dry_run:
        return DryRunAdapter()
    if provider == "xai":
        return XAIAdapter()
    if provider == "anthropic":
        return AnthropicAdapter()
    if provider == "openai":
        return OpenAIAdapter()
    if provider == "google":
        return GoogleAdapter()
    raise ValueError(f"unsupported provider {provider!r}")


def _as_int(value: object) -> int:
    """Coerce a usage field to a non-negative int."""
    if value is None:
        return 0
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        return max(0, int(value))
    return 0


def _attr_int(owner: object, name: str) -> int:
    """Read ``owner.name`` as a non-negative int."""
    return _as_int(getattr(owner, name, None))


class DryRunAdapter:
    """Deterministic offline adapter used by ``--dry-run`` and unit tests."""

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
        """Return a stable fake completion; ``timeout_s`` is accepted and ignored."""
        del timeout_s
        request_temperature = temperature if send_temperature else None
        text = f"DRY-RUN {model_id}\nFINAL: 1.0 m/s"
        usage = TokenUsage(
            input_tokens=max(1, len(system) + len(user)),
            output_tokens=max(1, len(text)),
            cached_input_tokens=0,
        )
        raw_request: dict[str, object] = {
            "model": model_id,
            "system": system,
            "user": user,
            "temperature": request_temperature,
            "max_output_tokens": max_output_tokens,
        }
        raw_response: dict[str, object] = {
            "id": "dry-run",
            "model": model_id,
            "text": text,
        }
        return AdapterResult(
            text=text,
            usage=usage,
            raw_request=raw_request,
            raw_response=raw_response,
            request_temperature=request_temperature,
            response_model=model_id,
            sdk_version="dry-run",
        )


class XAIAdapter:
    """xAI Chat Completions via the OpenAI-compatible SDK (no tools / no search)."""

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
        """Call chat completions with the canonical messages and no tools."""
        api_key = require_key("xai")
        client = OpenAI(api_key=api_key, base_url=XAI_BASE_URL, timeout=timeout_s)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        request_temperature = temperature if send_temperature else None
        raw_request: dict[str, object] = {
            "provider": "xai",
            "model": model_id,
            "messages": messages,
            "max_tokens": max_output_tokens,
            "temperature": request_temperature,
        }
        if send_temperature:
            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=max_output_tokens,
                temperature=temperature,
                timeout=timeout_s,
            )
        else:
            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=max_output_tokens,
                timeout=timeout_s,
            )
        usage = _usage_from_chat(response)
        echoed = getattr(response, "model", None)
        response_model = echoed if isinstance(echoed, str) else None
        return AdapterResult(
            text=_chat_text(response),
            usage=usage,
            raw_request=sanitize_object(raw_request),
            raw_response=sanitize_object(response),
            request_temperature=request_temperature,
            response_model=response_model,
            sdk_version=package_version("openai"),
            truncated=_chat_truncated(response),
        )


class OpenAIAdapter:
    """OpenAI Responses API. ``store=False``; tools omitted; no extra instructions."""

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
        """Call ``client.responses.create`` with canonical ``instructions`` and ``input``."""
        api_key = require_key("openai")
        client = OpenAI(api_key=api_key, timeout=timeout_s)
        request_temperature = temperature if send_temperature else None
        raw_request: dict[str, object] = {
            "provider": "openai",
            "model": model_id,
            "instructions": system,
            "input": user,
            "max_output_tokens": max_output_tokens,
            "store": False,
            "temperature": request_temperature,
        }
        if send_temperature:
            response = client.responses.create(
                model=model_id,
                instructions=system,
                input=user,
                max_output_tokens=max_output_tokens,
                store=False,
                temperature=temperature,
                timeout=timeout_s,
            )
        else:
            response = client.responses.create(
                model=model_id,
                instructions=system,
                input=user,
                max_output_tokens=max_output_tokens,
                store=False,
                timeout=timeout_s,
            )
        echoed = getattr(response, "model", None)
        response_model = echoed if isinstance(echoed, str) else None
        return AdapterResult(
            text=_openai_output_text(response),
            usage=_usage_from_responses(response),
            raw_request=sanitize_object(raw_request),
            raw_response=sanitize_object(response),
            request_temperature=request_temperature,
            response_model=response_model,
            sdk_version=package_version("openai"),
            truncated=_responses_truncated(response),
        )


class AnthropicAdapter:
    """Anthropic Messages API. System is a top-level parameter, not a message."""

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
        """Call ``messages.create``; skip thinking blocks when extracting ``text``."""
        api_key = require_key("anthropic")
        client = anthropic.Anthropic(api_key=api_key, timeout=timeout_s)
        request_temperature = temperature if send_temperature else None
        user_messages = [{"role": "user", "content": user}]
        raw_request: dict[str, object] = {
            "provider": "anthropic",
            "model": model_id,
            "system": system,
            "messages": user_messages,
            "max_tokens": max_output_tokens,
            "temperature": request_temperature,
        }
        if send_temperature:
            response = client.messages.create(
                model=model_id,
                system=system,
                messages=user_messages,
                max_tokens=max_output_tokens,
                temperature=temperature,
                timeout=timeout_s,
            )
        else:
            response = client.messages.create(
                model=model_id,
                system=system,
                messages=user_messages,
                max_tokens=max_output_tokens,
                timeout=timeout_s,
            )
        echoed = getattr(response, "model", None)
        response_model = echoed if isinstance(echoed, str) else None
        return AdapterResult(
            text=_anthropic_text(response),
            usage=_usage_from_anthropic(response),
            raw_request=sanitize_object(raw_request),
            raw_response=sanitize_object(response),
            request_temperature=request_temperature,
            response_model=response_model,
            sdk_version=package_version("anthropic"),
            truncated=_anthropic_truncated(response),
        )


class GoogleAdapter:
    """Gemini ``generate_content`` with BLOCK_NONE safety (aerospace stems)."""

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
        """Call Gemini Developer API; safety BLOCK_NONE is adapter-only, not prompt text."""
        api_key = require_key("google")
        timeout_ms = int(timeout_s * 1000)
        client = genai.Client(
            api_key=api_key,
            http_options=genai_types.HttpOptions(timeout=timeout_ms),
        )
        request_temperature = temperature if send_temperature else None
        safety = [
            genai_types.SafetySetting(
                category=category,
                threshold=genai_types.HarmBlockThreshold.BLOCK_NONE,
            )
            for category in _GOOGLE_HARM_CATEGORIES
        ]
        if send_temperature:
            config = genai_types.GenerateContentConfig(
                system_instruction=system,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                safety_settings=safety,
            )
        else:
            config = genai_types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=max_output_tokens,
                safety_settings=safety,
            )
        raw_request: dict[str, object] = {
            "provider": "google",
            "model": model_id,
            "system_instruction": system,
            "contents": user,
            "temperature": request_temperature,
            "max_output_tokens": max_output_tokens,
            "safety_settings": "BLOCK_NONE",
        }
        response = client.models.generate_content(
            model=model_id,
            contents=user,
            config=config,
        )
        echoed = getattr(response, "model_version", None)
        response_model = echoed if isinstance(echoed, str) else None
        text = getattr(response, "text", None)
        visible = text if isinstance(text, str) else ""
        return AdapterResult(
            text=visible,
            usage=_usage_from_google(response),
            raw_request=sanitize_object(raw_request),
            raw_response=sanitize_object(response),
            request_temperature=request_temperature,
            response_model=response_model,
            sdk_version=package_version("google-genai"),
            truncated=_google_truncated(response),
        )


def _chat_truncated(response: object) -> bool:
    """True when a Chat Completions response stopped at the token cap."""
    choices = getattr(response, "choices", None)
    if not isinstance(choices, list) or len(choices) == 0:
        return False
    return getattr(choices[0], "finish_reason", None) == "length"


def _responses_truncated(response: object) -> bool:
    """True when an OpenAI Responses call ended incomplete on max_output_tokens."""
    if getattr(response, "status", None) != "incomplete":
        return False
    details = getattr(response, "incomplete_details", None)
    return getattr(details, "reason", None) == "max_output_tokens"


def _anthropic_truncated(response: object) -> bool:
    """True when an Anthropic message stopped with ``stop_reason == max_tokens``."""
    return getattr(response, "stop_reason", None) == "max_tokens"


def _google_truncated(response: object) -> bool:
    """True when the first Gemini candidate finished on MAX_TOKENS."""
    candidates = getattr(response, "candidates", None)
    if not isinstance(candidates, list) or len(candidates) == 0:
        return False
    reason = getattr(candidates[0], "finish_reason", None)
    if reason is None:
        return False
    name = getattr(reason, "name", None)
    text = name if isinstance(name, str) else str(reason)
    return text.endswith("MAX_TOKENS")


def _chat_text(response: object) -> str:
    """Extract assistant text from a Chat Completions response."""
    choices = getattr(response, "choices", None)
    if not isinstance(choices, list) or len(choices) == 0:
        return ""
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    return ""


def _usage_from_chat(response: object) -> TokenUsage:
    """Map OpenAI/xAI Chat Completions usage onto ``TokenUsage``."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return TokenUsage()
    details = getattr(usage, "prompt_tokens_details", None)
    cached = _attr_int(details, "cached_tokens") if details is not None else 0
    return TokenUsage(
        input_tokens=_attr_int(usage, "prompt_tokens"),
        output_tokens=_attr_int(usage, "completion_tokens"),
        cached_input_tokens=cached,
    )


def _openai_output_text(response: object) -> str:
    """Extract visible text from a Responses API object; skip reasoning items."""
    direct = getattr(response, "output_text", None)
    if isinstance(direct, str) and direct:
        return direct
    chunks: list[str] = []
    output = getattr(response, "output", None)
    if not isinstance(output, list):
        return ""
    for item in output:
        item_type = getattr(item, "type", None)
        if item_type == "reasoning":
            continue
        content = getattr(item, "content", None)
        if not isinstance(content, list):
            continue
        for part in content:
            part_type = getattr(part, "type", None)
            if part_type in ("output_text", "text"):
                text = getattr(part, "text", None)
                if isinstance(text, str):
                    chunks.append(text)
            elif part_type == "refusal":
                refusal = getattr(part, "refusal", None)
                if isinstance(refusal, str):
                    chunks.append(refusal)
    return "".join(chunks)


def _usage_from_responses(response: object) -> TokenUsage:
    """Map OpenAI Responses usage onto ``TokenUsage``."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return TokenUsage()
    details = getattr(usage, "input_tokens_details", None)
    cached = _attr_int(details, "cached_tokens") if details is not None else 0
    return TokenUsage(
        input_tokens=_attr_int(usage, "input_tokens"),
        output_tokens=_attr_int(usage, "output_tokens"),
        cached_input_tokens=cached,
    )


def _anthropic_text(response: object) -> str:
    """Concatenate ``text`` content blocks; skip thinking / tool blocks."""
    chunks: list[str] = []
    content = getattr(response, "content", None)
    if not isinstance(content, list):
        return ""
    for block in content:
        if getattr(block, "type", None) != "text":
            continue
        text = getattr(block, "text", None)
        if isinstance(text, str):
            chunks.append(text)
    return "".join(chunks)


def _usage_from_anthropic(response: object) -> TokenUsage:
    """Map Anthropic usage; cache reads count as cached input tokens."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return TokenUsage()
    return TokenUsage(
        input_tokens=_attr_int(usage, "input_tokens"),
        output_tokens=_attr_int(usage, "output_tokens"),
        cached_input_tokens=_attr_int(usage, "cache_read_input_tokens"),
    )


def _usage_from_google(response: object) -> TokenUsage:
    """Map Gemini usage; thoughts tokens are billed as output."""
    meta = getattr(response, "usage_metadata", None)
    if meta is None:
        return TokenUsage()
    output_tokens = _attr_int(meta, "candidates_token_count") + _attr_int(
        meta, "thoughts_token_count"
    )
    return TokenUsage(
        input_tokens=_attr_int(meta, "prompt_token_count"),
        output_tokens=output_tokens,
        cached_input_tokens=_attr_int(meta, "cached_content_token_count"),
    )
