# Max-Q

**An independent aerospace engineering benchmark for frontier LLMs.**

Propulsion, orbital mechanics, structural analysis, GNC, telemetry interpretation — scored, reproducible, no affiliations.

## Why

Grok 4.7 claims supplemental training on decades of internal SpaceX engineering data. That claim is unverified. Max-Q tests whether it (and any frontier model) can actually do aerospace engineering — against Claude, GPT, and Gemini — with published methodology, published scoring code, and published results.

No lab funding. No aerospace employer. No axe to grind. Just the harness, the questions, and the numbers.

## How it works

1. **Question sets are held out.** Each question wave is authored privately. Before any model is run, the SHA-256 hash of the frozen question file is committed to `questions/HASHES.md` (and posted publicly). After all models are scored, the questions are published and anyone can verify the hash — proving no post-hoc tuning by us or anyone else.
2. **Every model gets identical treatment.** Same prompts, same temperature, same number of attempts, same scoring rubric. All run transcripts are published.
3. **Scoring is mechanical wherever possible.** Numeric answers with tolerance bands, unit-checked; derivations rubric-scored with the rubric published; telemetry tasks scored against ground truth from public datasets.

## Domains

| Domain | Examples |
|---|---|
| Propulsion | nozzle flow, staging optimization, engine cycle analysis |
| Orbital mechanics | transfers, rendezvous, perturbations, constellation geometry |
| Structures | loads, margins, buckling, pressure vessels |
| GNC | attitude dynamics, control margins, sensor fusion |
| Telemetry | anomaly detection and interpretation on real public telemetry |

## Repo layout

- `maxq/` — run harness and scoring code
- `questions/` — hash commitments, then published question waves (post-run)
- `results/` — per-model transcripts and scored results
- `docs/` — methodology, rubric, contamination policy

## Status

Pre-launch. Wave 1 in development. Grok 4.7 not yet released; harness is validated against Grok 4.6 as a stand-in.

## License

MIT for all code. Question sets released CC BY 4.0 after each wave is scored.
