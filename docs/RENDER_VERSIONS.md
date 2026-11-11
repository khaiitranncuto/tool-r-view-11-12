# Render versions and rollback

Update: by user request, both render paths now extend source END as far as
needed for speech, capped at the next **locked** source-shot start or source EOF.
This is not scene-cut detection: an unindexed cut inside that gap is not known
to the renderer. Legacy rendering can still hold the remaining frame if speech
exceeds that cap; v2 reports insufficient footage. The strict planner default
remains unchanged for callers that do not opt in. The v2 video-cache version was
bumped so old frozen segments cannot be reused under the new policy. The frozen
legacy snapshot is unchanged.

The default remains `legacy_v1` (Render cũ). Select `structured_v2` (Render mới)
in the export step to try the new MP4 renderer. There is no silent fallback:
if v2 fails, select Render cũ and submit again. CapCut export uses v1.

## Frozen snapshot

`.render_backups/legacy_before_structured_v2/manifest.json` records SHA-256 hashes
of the pre-v2 app, renderer, TTS, worker, encoding, guard, template and JS files.
The snapshot preserves the situation-recap/character improvements already present
when the new render work started.

To restore those exact files, stop the app and active render workers, then run:

```powershell
python .render_backups/legacy_before_structured_v2/restore.py
```

Restart the app. Restoration verifies every backup first and does not delete
projects, input/output videos or TTS cache. `structured_render.py` and this document
can remain on disk; the restored app does not invoke v2.

## Data flow

V1: editable TIME/BEAT strings -> parser -> grouped TTS -> word-led cuts ->
encoded segments -> concatenation -> subtitles and MP4. It retains its original
retry defaults and temporary-file cleanup.

V2: verified analysis JSON -> validated versioned situation plan -> paragraph
TTS -> exact word-cue cut positions -> locked source shots -> segments -> MP4.
The editor's TIME/BEAT view is validated once at the request boundary; the worker
does not reparse that text. Edited narration is accepted only if the selected
source groups and cut anchors still agree. The originating analysis task must
match the project and video. Missing data never triggers a new Gemini analysis.

V2 uses 2 TTS connections, starts at least 4 seconds apart, with at most 3 attempts
per request. It has four total rounds: initial, recovery, recovery, 180-second wait
then final recovery. Recovery is sequential and only handles failed groups. The
last-resort error-triggered splitting is disabled; text exceeding the service's
safe length still splits normally. Existing successful audio cache is reused
when text/voice/rate match.

V2 stages and segment-completion markers live in `.render_cache/<fingerprint>`.
The fingerprint includes the JSON plan, source/BGM path-size-mtime, voice/rate and
render settings. Failed jobs retain the cache. V1 and v2 do not share video-stage
cache. V2 refuses missing timing, overlapping ranges and insufficient footage
instead of estimating cuts, borrowing unselected frames or creating long holds.
It shares the low-level encoder and segment machinery with v1, with segment
resume enabled only for v2. No claim of a speedup or improved service availability
is made without an end-to-end run.
