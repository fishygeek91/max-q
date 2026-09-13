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
- Numeric: unit-aware (pint), relative tolerance declared per question at freeze time.
- Derivations: published rubric, LLM-assisted first pass, human-confirmed.
- All disagreements and judgment calls logged publicly.

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
- **Models (pinned ids):** `grok-4.6` (permanent 4.6 baseline),
  `claude-fable-5-1`, `gpt-6-astra`, `gemini-3.1-pro` (preview flagship),
  `gemini-3.8-flash` (small-model floor). A later Grok generation is a new
  config row. Transcripts store `requested_model`, `response_model`, and UTC
  `started_at` / `finished_at`.
- **Resume:** skip a triple only when a valid transcript exists and `error` is
  null. `cost.json` is rebuilt from transcripts on disk after every write.

Scoring later reports both pass@1 and best-of-n. Full transcripts are published.

## Credibility protocol
SHA-256 of the frozen question file is committed publicly BEFORE any model runs
(questions/HASHES.md + public post). Questions published after scoring. Code MIT,
questions CC BY 4.0. No funding from, or affiliation with, any AI lab or
aerospace company.
