# Freeze → run → publish checklist

Mechanical sequence for a question wave. Do not improvise under launch-day
time pressure. Hashing is SHA-256 of the private JSON **as stored on disk** —
never re-serialize, pretty-print, or sort keys after freeze.

Wave 1 is frozen to `f7b1f78c…` (39211 bytes) and posted
([issue #5 comment](https://github.com/fishygeek91/max-q/issues/5#issuecomment-5667748349);
voided first digest documented in `questions/HASHES.md`). Remaining for Wave 1:
run enabled rows → Claude first-pass judge → Darth confirm → later Grok
generation → publish (steps 5–7). Do not publish between the 4.6 baseline and
the next Grok row.

## 1. Author and lock bytes

1. Keep the wave in `questions/private/` (gitignored).
2. Lock JSON layout once with `dump_questions` (UTF-8, Unix newlines, indent 2).
3. After this point, **never rewrite** the file. Inventory-only checks:
   `python -m maxq.wave questions/private/wave-N.json`.

## 2. Freeze (hash commitment)

```
python scripts/freeze_wave.py --wave N --questions questions/private/wave-N.json
python scripts/freeze_wave.py --wave N --questions questions/private/wave-N.json --run
```

Dry-run prints the SHA-256, byte count, proposed HASHES.md row, and a public-post
template. `--run` appends the row. Re-running on a matching file is a no-op;
a different digest is refused (exit 2).

## 3. Commit the commitment only

Commit `questions/HASHES.md` only. Do not add `questions/private/`.

## 4. Post the hash publicly

Paste the freeze script's public-post template to:

1. A GitHub issue comment (server timestamp the repo owner cannot rewrite)
2. A gist and/or X post (second independent timestamp)

Then replace the HASHES.md **Posted** em dash with those URL(s). Do not start
model runs until Posted is filled.

## 5. Run, then score

```
python -m maxq.runner --wave N --questions questions/private/wave-N.json
python -m maxq.scoring --wave N --questions questions/private/wave-N.json --judge-model anthropic/claude-fable-5-1
python scripts/render_rubric_review.py --wave N --questions questions/private/wave-N.json
python -m maxq.scoring --wave N --questions questions/private/wave-N.json \
  --accept-from results/wave-N/accept.json --apply-overrides results/wave-N/human-overrides.json \
  --reviewer Darth
python scripts/wave_metrics.py --wave N
```

Use the committed `config/run.json` (every `enabled: true` row). Do not
`--accept-llm` on a wave that will be published. Do not publish between the
Grok 4.6 baseline and a later Grok generation — that leaks stems and voids
the delta. `freeze_wave.py --run` cannot supersede a conflicting HASHES.md
row; a voided freeze is a manual HASHES.md edit plus a new public post, and
only if no model has been run.

## 6. Publish

```
python scripts/publish_wave.py --wave N --questions questions/private/wave-N.json
python scripts/publish_wave.py --wave N --questions questions/private/wave-N.json --run
```

Refuses unless every enabled model has complete transcripts **and** the file
still matches the frozen hash. `--run` copies to `questions/wave-N.json` and
`git add -f`s gitignored `results/wave-N/`. Set **Published** to that file.

## 7. CI

Push. CI runs pytest, including `test_published_waves_match_hashes`: every
tracked `questions/wave-*.json` must equal its HASHES.md SHA-256.
