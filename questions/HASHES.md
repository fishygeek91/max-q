# Question wave hash commitments

Before any model is run on a wave, the frozen question file's SHA-256 is recorded
here (and posted publicly). After all models are scored, the file is published to
this directory and anyone can verify the hash.

| Wave | Frozen (UTC) | SHA-256 | Posted | Published |
|---|---|---|---|---|
| 1 | 2026-09-13T15:22:48Z | f7b1f78c5e35c4b2d58472bf361d6e117980ed3bbde358d89b7229336872bc5e | [issue #5 comment](https://github.com/fishygeek91/max-q/issues/5#issuecomment-5667748349) | — |

**Posted** = external, server-timestamped record of the commitment made before any
model run (a GitHub issue comment cannot be rewritten by the repo owner the way git
history can; add an X post or gist as a second venue when possible). **Published** =
link to the question file released after all enabled models are scored, via
`scripts/publish_wave.py`.

## Wave 1 algorithm

Wave 1's commitment is SHA-256 of `questions/private/wave-1.json` **as stored on disk**
(UTF-8, 39211 bytes). Do not re-serialize, pretty-print, or sort keys before hashing.
`scripts/freeze_wave.py` hashes those exact bytes.

## Wave 1 supersede (before any model run)

The first freeze is **void**. No model was run against it.

- Voided SHA-256: `0c38cba1425bfe22ddb9698b5d9a8bf634f20c4d6e65ca41e66b9a393b8e9e24`
  Frozen (UTC) 2026-09-13T14:38:08Z, 39117 bytes. Public post:
  [issue #11 comment](https://github.com/fishygeek91/max-q/issues/11#issuecomment-5655494678).
- Reason: `W1-PROP-005` used an unphysical throat area (implied `c*` efficiency
  well above 1); `W1-TEL-010` no longer labels an arbitrary epoch as a close
  approach. See [issue #1](https://github.com/fishygeek91/max-q/issues/1).
- Current SHA-256: `f7b1f78c5e35c4b2d58472bf361d6e117980ed3bbde358d89b7229336872bc5e`
  Frozen (UTC) 2026-09-13T15:22:48Z, 39211 bytes.
