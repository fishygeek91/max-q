# Question wave hash commitments

Before any model is run on a wave, the frozen question file's SHA-256 is recorded
here (and posted publicly). After all models are scored, the file is published to
this directory and anyone can verify the hash.

| Wave | Frozen (UTC) | SHA-256 | Published |
|---|---|---|---|
| 1 | 2026-09-13T15:22:48Z | f7b1f78c5e35c4b2d58472bf361d6e117980ed3bbde358d89b7229336872bc5e | — |

## Wave 1 algorithm

Wave 1's commitment is SHA-256 of `questions/private/wave-1.json` **as stored on disk**
(UTF-8, 39211 bytes). Do not re-serialize, pretty-print, or sort keys before hashing.
Issue #4 freeze tooling must hash those exact bytes.

## Wave 1 superseded freeze

The first freeze (`2026-09-13T14:38:08Z`, SHA-256
`0c38cba1425bfe22ddb9698b5d9a8bf634f20c4d6e65ca41e66b9a393b8e9e24`, 39117 bytes)
is **void**. Before any model run, `W1-PROP-005` was found to use an unphysical
throat area (implied `c*` efficiency well above 1). The throat area was resized
and `W1-TEL-010` was reworded so the stated epoch is not labeled a close approach.
No model runs were performed against either digest.
