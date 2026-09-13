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
Same prompt template, temperature, attempt count for every model. Best-of-n and
pass@1 both reported. Full transcripts published.

## Credibility protocol
SHA-256 of the frozen question file is committed publicly BEFORE any model runs
(questions/HASHES.md + public post). Questions published after scoring. Code MIT,
questions CC BY 4.0. No funding from, or affiliation with, any AI lab or
aerospace company.
