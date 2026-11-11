# Source-grounded writing — evidence contract 19

This replaces the v18 situation workflow for newly analyzed projects. Legacy
projects retain their existing path. Render rollback files are unchanged.

## One writing request, then backend binding

V4 situation plans carry `source_link_contract_version=1`. The existing writing
request receives real source frames, scene-local filtered Whisper, stable fact
IDs, character context and the previous narrative tail. It returns individual
sentences, their fact IDs, a primary visual ID and visual/dialogue evidence kind.
There is no separate model request to retype the locked prose and map pictures,
and no independent whole-script video audit on this new path.

Backend checks exact spoken sentence totals, fact/visual membership, evidence
kind and source chronology. It generates literal TTS anchors from the frozen
sentences. Excess sentence boundaries may be joined without changing words;
shortages are never padded. Failed candidates are retained, the existing bounded
retry receives the previous draft, and a worse retry cannot overwrite a usable
earlier result. No retry limits have increased.

The method is explicitly stored as `source_grounded_generation`, not an
independent `video` verification. Its signature binds exact narration, sentence
proofs, fact records and selected source IDs across serialization. This checks
data integrity, not whether every AI interpretation of a picture is correct.

## Source geometry and clip reuse

Detailed clips retain the original timeline and cold-open protections. They are
cached with source stat, interval, sampling configuration and output size, so an
API-key switch does not encode identical clips again. Changing source or config
invalidates the cache. Nonmatching or truncated outputs are regenerated.

V4 locking supplies every proof window available inside its situation, capped
before the next situation. Fact references are restricted to the actual supplied
proof. Full-situation writing packs use the whole situation pool. Export packs
may select representative pictures but preserve the original V IDs; V005 does
not become V003 after filtering. Backend source IDs and ranges remain authoritative.

## Characters and dialogue

Writing context separates visible people from people mentioned in facts or
dialogue. Both may legally be named in narration; a referenced person is not
automatically treated as visible. Common phrases `thời gian` and `không gian`
do not count as references to the character Gian.

Unknown people retain stable canonical IDs and descriptive aliases. A later
supported name can update the same ID. Generic alias equality alone is no
longer enough to merge different appearances: compatible visual features are
required. The film registry propagates later confirmed names.

ASR preprocessing removes promotional material, long sparse recognition noise
and suspicious long repeated phrases before feeding Gemini. It does not ban a
language globally or request new transcription. Fact-level timestamp overlap
still controls which dialogue accompanies each situation.

## Validation and limits

- Full offline suite: 518 tests, 5 skipped; all remaining tests passed.
- Actual archived ec396 replay: 134 situations, exactly 281 allocated sentences,
  zero missing fact-to-picture references in all three writing contexts.
- The replay produces 45 visual-writing batches (19/15/11). This is a planned
  request count, not a measured run time or a guarantee against quota/timeouts.
- Nested real app writer tested with offline model responses: one request per
  visual batch and zero additional mapping requests; resume skips completed work.
- Real FFmpeg tests cover short MP4 export and indexed-clip reuse without recuts.
- No new full-film Gemini/Groq run or full movie export was performed.

Gemini can still misinterpret imagery. Unsupported details must be omitted at
writing time; the backend does not invent evidence, identities or filler to force
success. Exact sentence count remains mandatory; words remain reference-only.
