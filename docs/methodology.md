# Methodology (draft)

## Question sourcing & contamination policy
Questions are authored fresh — never copied from textbooks, problem sets, or public
benchmarks. Contamination defenses, in priority order:
1. Novel parameterization: standard physics, non-standard numbers/configurations,
   so a memorized worked example gives the wrong answer.
2. Composite problems: chain 2-3 domains (e.g. a staging change that alters both
   delta-v budget and structural margin) — composition is where memorization fails.
3. Real-data tasks: telemetry interpretation on public datasets (e.g. public
   Starlink TLEs, CelesTrak, launch webcast-derived data) with questions whose
   answers require computation, not recall.
4. Each question carries a `provenance` note describing how it was authored.

## Difficulty calibration
Three tiers: undergrad (any strong model should pass — sanity floor),
practitioner (working-engineer tasks), expert (edge cases, real trade studies).
A model's headline score is meaningless without the tier breakdown; we publish all.

## Scoring

Mechanical wherever possible. The CLI is `python -m maxq.scoring`.

- **Extract:** the payload of the last `FINAL:` line in `Transcript.text`
  (case-insensitive). Missing or empty FINAL is unparseable.
- **Numeric (`numeric_tolerance`):** pint-parse the payload (bare numbers inherit
  the question unit), convert into that unit, and pass iff
  `abs(candidate - truth) <= rel_tol * abs(truth)` (boundary inclusive). A zero
  truth matches only a converted zero. Unparseable or incompatible units score 0
  with a scorer note.
- **Series (`ground_truth_series`):** the FINAL payload must be a JSON array of
  the same length as the frozen precomputed truth in `Question.answer`. Each
  element is converted into the question unit and checked with the same
  `rel_tol`. Truth is *not* recomputed at score time (poliastro/astropy stay in
  the optional `verify` extra used when authoring items).
- **Exact:** stripped, case-sensitive string equality on the FINAL payload.
- **Truncation:** if `truncated` is true and the FINAL line is missing or
  unparseable, the attempt is `truncated` with `score=null` — not incorrect. A
  truncated attempt that still has a parseable FINAL is scored normally and
  noted.
- **Rubric:** LLM-assisted first pass (`--judge-model` / `--judge-stub`) writes
  `results/wave-N/rubric-queue.json`. Official scores stay `pending_rubric` until
  `--accept-llm` (confirm the first pass as-is) or `--apply-overrides PATH`.
  Every accept and every criterion change is appended to
  `results/wave-N/overrides.jsonl`. Pass@1 / best-of-n require every bullet
  confirmed true (`score == 1.0`); pending rows cannot pass.
- **Reporting:** `results/wave-N/scored.json` plus a model×tier table
  (undergrad / practitioner / expert) with pass@1, best-of-n, truncated,
  unparseable, and pending-rubric counts. Never a single headline number.
- **Human confirm:** `--accept-from PATH` confirms listed
  `(question_id, model, attempt)` triples as-is without changing which
  models are scored. Blanket `--accept-llm` is for smoke/fixtures only.

## Identical treatment
Every model gets the same closed-book treatment. The committed spec is
`config/run.json` (template id `maxq-closed-book-v1`).

- **Prompt:** one system prompt + the question stem + a shared `FINAL:` suffix.
  Canonical text is stored on every transcript as `rendered_messages`. Wire
  shape differs by vendor (Anthropic `system=`, Gemini `system_instruction`,
  OpenAI Responses `instructions=`); the text does not.
- **Sampling:** `temperature` 0.0, `attempts` 3 as **three independent API
  calls** (never a vendor batch `n=`). Some reasoning models reject
  `temperature`; those rows set `send_temperature: false`. Transcripts record
  both `configured_temperature` and `request_temperature`.
- **Tools:** none. No web search. OpenAI Responses uses `store=false` so
  held-out stems are not kept in provider prompt history.
- **Gemini safety:** the Google adapter sets `BLOCK_NONE` so propulsion/GNC
  items are not pre-filtered. This is not jailbreak text in the shared prompt.
  A model refusal (HTTP 200) is a completed attempt.
- **Models (pinned ids):** OpenRouter slugs `x-ai/grok-4.6` (permanent 4.6
  baseline), `anthropic/claude-fable-5-1`, `openai/gpt-6-astra`,
  `google/gemini-3.1-pro-preview` (preview flagship; the unsuffixed
  `google/gemini-3.1-pro` is not a live OpenRouter id), `google/gemini-3.8-flash`
  (small-model floor). Google rows pin `openrouter_providers: ["Google AI Studio"]`
  (OpenRouter's first-party Gemini API name; `Google` is Vertex). A later Grok
  generation is a new config row. Transcripts store `requested_model`,
  `response_model`, `served_by`, and UTC `started_at` / `finished_at`.
- **Resume:** skip a triple only when a valid transcript exists and `error` is
  null. `cost.json` is rebuilt from transcripts on disk after every write.

Scoring reports both pass@1 and best-of-n per model and per tier. Full
transcripts are published.

## Credibility protocol
SHA-256 of the frozen question file is committed publicly BEFORE any model runs
(questions/HASHES.md + public post). Freeze, post, run, and publish are the
checklist in `docs/freeze-run-publish.md` (`scripts/freeze_wave.py`, then
`scripts/publish_wave.py`). Questions published after scoring. Code MIT,
questions CC BY 4.0. No funding from, or affiliation with, any AI lab or
aerospace company.

## Output-token budgets and truncation

Reasoning models spend tokens on internal thinking. On Gemini, thinking tokens
count against the request's visible-output cap, so Google model rows carry a
documented `max_output_tokens` override in `config/run.json`; all other
providers use the run-level budget. Every adapter records whether the provider
stopped at the token cap (`truncated: true` in the transcript). A truncated
attempt is reported separately from an incorrect one — an absent FINAL line at
the cap is a budget artifact, not evidence about the model's engineering
ability. `--verify-models` pings with a fixed 1024-token budget and passes on
any successful response, even with empty visible text.

## Publication gate

Wave transcripts embed the question stems, so `results/wave-*/` is gitignored
and a wave is published only through `scripts/publish_wave.py`, which refuses
until every enabled model in the run config has complete, error-free
transcripts and the question file still matches its frozen hash. In
particular: after the Grok 4.6 baseline run, the wave stays private until the
next Grok generation has been run and scored — publishing in between would let
the newer model see the questions and void the delta.

## API gateway

All contestant calls are routed through OpenRouter with one operator key.
Fallback routing is disabled on every request (`provider.allow_fallbacks:
false`, plus an explicit first-party `order` where pinned in the config), so a
call is served by the intended upstream or fails loudly. Google rows pin
`Google AI Studio` (OpenRouter's first-party Gemini API `provider_name`);
`Google` is Vertex and is not used. The serving provider OpenRouter reports is
persisted in every transcript as `served_by` and is part of the published audit
trail. Direct vendor adapters remain in the codebase as a fallback path.
