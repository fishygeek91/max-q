# Wave 1 rehearsal notes (issue #5)

Stem-free log of the pre-4.7 rehearsal. Transcripts stay gitignored under
`results/wave-1/`. Do not publish until a later Grok generation has been scored
(`scripts/publish_wave.py`).

## Freeze

- Current SHA-256: `f7b1f78c5e35c4b2d58472bf361d6e117980ed3bbde358d89b7229336872bc5e`
- Frozen (UTC): 2026-09-13T15:22:48Z
- Bytes: 39211
- Posted: [issue #5 comment](https://github.com/fishygeek91/max-q/issues/5#issuecomment-5667748349)
- Voided first digest (no model runs): `0c38cba1425bfe22ddb9698b5d9a8bf634f20c4d6e65ca41e66b9a393b8e9e24`
  (39117 bytes, 2026-09-13T14:38:08Z, [issue #11](https://github.com/fishygeek91/max-q/issues/11#issuecomment-5655494678))
- `python scripts/freeze_wave.py --wave 1 --questions questions/private/wave-1.json` → `HASHES.md: verified`

## Enabled run set

Derived only from `enabled: true` in `config/run.json` (do not hardcode a count):

- `grok-4.6` (baseline)
- `claude-fable-5-1` (flagship; rubric judge for the first pass)
- `gemini-3.1-pro` (flagship)
- `gemini-3.8-flash` (small_floor)

`gpt-6-astra` stays `enabled: false` (no API access for this operator). The
writeup must say OpenAI's flagship was not tested.

## Smoke (no Wave 1 contamination)

- Full suite: 75 passed after harness fixes (baseline on 2586b84 was 69 passed, 2 failed)
- `ruff check maxq tests scripts` clean
- Fixture dry-run: `python -m maxq.runner --questions tests/fixtures/sample_wave.json --wave smoke --dry-run`
- Fixture `--judge-stub --accept-llm` on `tests/fixtures/scoring_wave.json` with `--wave smoke` (never on wave-1)

## Live run

**Not started.** `python -m maxq.runner --verify-models` exited 1 in this
environment: `XAI_API_KEY`, `ANTHROPIC_API_KEY`, and `GOOGLE_API_KEY` /
`GEMINI_API_KEY` were unset. Resume the paid wave locally after exporting those
three keys:

```
python -m maxq.runner --verify-models
python -m maxq.runner --wave 1 --questions questions/private/wave-1.json
python -m maxq.scoring --wave 1 --questions questions/private/wave-1.json --judge-model claude-fable-5-1
python scripts/render_rubric_review.py --wave 1 --questions questions/private/wave-1.json
```

Do not `--accept-llm` on Wave 1. Darth confirms the 96 rubric rows via
`--accept-from` / `--apply-overrides --reviewer Darth`. Then:

```
python scripts/wave_metrics.py --wave 1
python scripts/publish_wave.py --wave 1 --questions questions/private/wave-1.json
```

The publish dry-run must pass the completeness + hash gate; do not pass `--run`
until a later Grok generation is scored.

Extraction rate, rubric-agreement share, USD, and wall-clock will land in a
follow-up edit of this file after the live wave (from `scored.json`,
`cost.json`, `overrides.jsonl`, and transcript timestamps).

## Harness fixes found during rehearsal

1. **Enabled-row tests lagged the Astra disable.** `test_config_pins_expected_models`
   and the dry-run provider set still required `gpt-6-astra` / `openai` after
   commit `2586b84`. Tests now pin every config row and derive the run set from
   `enabled`.
2. **`--accept-llm` plus `--model` would drop contestants from `scored.json`.**
   Added `--accept-from` so a human can confirm listed triples while still scoring
   every enabled row.
3. **No grouped rubric review surface.** Added `scripts/render_rubric_review.py`
   (stems; gitignored output) with `SELF-JUDGE` when `judge_model == model`.
4. **No stem-free wall-clock / extraction rollup.** Added `scripts/wave_metrics.py`.
5. **HASHES.md still listed the voided first digest** while the private file
   matched the corrected freeze. Superseded in HASHES.md (freeze_wave.py cannot
   overwrite a conflicting row). Local private-file test now asserts SHA-256
   equality when the file is present.
6. Open [PR #8](https://github.com/fishygeek91/max-q/pull/8) closed as superseded
   by this branch.
