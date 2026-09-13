# Question wave hash commitments

Before any model is run on a wave, the frozen question file's SHA-256 is recorded
here (and posted publicly). After all models are scored, the file is published to
this directory and anyone can verify the hash.

| Wave | Frozen (UTC) | SHA-256 | Published |
|---|---|---|---|
| 1 | 2026-09-13T14:38:08Z | 0c38cba1425bfe22ddb9698b5d9a8bf634f20c4d6e65ca41e66b9a393b8e9e24 | — |

## Wave 1 algorithm

Wave 1's commitment is SHA-256 of `questions/private/wave-1.json` **as stored on disk**
(UTF-8, 39117 bytes). Do not re-serialize, pretty-print, or sort keys before hashing.
Issue #4 freeze tooling must hash those exact bytes.
