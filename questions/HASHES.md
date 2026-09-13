# Question wave hash commitments

Before any model is run on a wave, the frozen question file's SHA-256 is recorded
here (and posted publicly). After all models are scored, the file is published to
this directory and anyone can verify the hash.

| Wave | Frozen (UTC) | SHA-256 | Posted | Published |
|---|---|---|---|---|
| 1 | 2026-09-13T14:38:08Z | 0c38cba1425bfe22ddb9698b5d9a8bf634f20c4d6e65ca41e66b9a393b8e9e24 | [issue #11 comment](https://github.com/fishygeek91/max-q/issues/11#issuecomment-5655494678) | — |

**Posted** = external, server-timestamped record of the commitment made before any
model run (a GitHub issue comment cannot be rewritten by the repo owner the way git
history can; add an X post or gist as a second venue when possible). **Published** =
link to the question file released after all enabled models are scored, via
`scripts/publish_wave.py`.

## Wave 1 algorithm

Wave 1's commitment is SHA-256 of `questions/private/wave-1.json` **as stored on disk**
(UTF-8, 39117 bytes). Do not re-serialize, pretty-print, or sort keys before hashing.
Issue #4 freeze tooling must hash those exact bytes.
