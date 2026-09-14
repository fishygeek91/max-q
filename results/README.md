# Results

Per-wave, per-model: full transcripts, scored JSON, and the summary table used
in the published writeup.

## Transcript layout

```
results/wave-N/<model-id>/{question_id}-a{attempt}.json
results/wave-N/cost.json
```

`--wave 1` writes `wave-1/`; `--wave smoke` writes `wave-smoke/` (gitignored).
Attempt filenames are `a1`, `a2`, `a3`. Model ids with `/` become `--`.

Each transcript is one `(question, model, attempt)` triple: canonical
`rendered_messages`, full raw request/response (JSON, credentials redacted),
visible `text`, token usage, `cost_usd`, `requested_model`, `response_model`,
and UTC timestamps. Answer keys, provenance, and API keys are never persisted.

## Scoring layout

```
results/wave-N/scored.json
results/wave-N/rubric-queue.json
results/wave-N/overrides.jsonl
```

`python -m maxq.scoring` reads transcripts and the question file (keys stay in
the question file; they are not copied back into transcripts). `scored.json` is
a `WaveScoreReport`: per-attempt rows plus per-model pass@1 / best-of-n with a
mandatory undergrad / practitioner / expert breakdown. Rubric items are first
scored by an injectable judge, then confirmed via `--accept-llm` or
`--apply-overrides`; the override log is append-only.

## Resume

A triple is skipped only when the file parses as a `Transcript` and `error`
is null. Corrupt JSON or a recorded error is retried and overwritten. Writes
use a `.tmp` file, fsync, then replace.

## Cost

`cost.json` is **rebuilt from transcripts on disk** after every persist
(never a running total). Rates come from `config/run.json`
(`pricing_usd_per_mtok`). Dry-run still writes fake usage so the path is
tested.

**Publication gate:** `results/wave-*/` is gitignored (transcripts embed
held-out stems). Publish only via `scripts/publish_wave.py` after every
enabled model is scored — see docs/methodology.md.
