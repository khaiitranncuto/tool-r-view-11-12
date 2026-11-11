# Reliability review — 2026-11-09

Reviewed the analysis, writing, normalization, task persistence, browser polling,
TTS recovery, encoder fallback and structured export paths. No production Gemini,
Groq or Edge TTS requests were made during verification.

## Fixes

- Recognize interrupted HTTP streams by exception type, cause chain and message.
  Retry the current request once even when its ordinary attempt budget is one.
  Discard partial text before retrying. Keep quota/auth failures separate and
  let the story-planning caller retain ownership of its own retry policy.
- Key the observation cache by the original video's identity, subtitle input,
  evidence version and FPS, rather than a newly generated temporary upload path.
- Persist terminal task states immediately even if progress was saved less than
  15 seconds earlier. Record the analysis process ID and report an interrupted
  analysis after its process exits, preserving checkpoints.
- Exclude internal writing caches and checkpoints from status responses from both
  RAM and disk. They remain stored for recovery.
- Record render worker startup/import/payload failures before the heartbeat starts.
- Clean up temporary checkpoint files after serialization or replacement failures
  while preserving the previous checkpoint.
- Validate API JSON object shape, string fields and finite numeric options.
  Reject invalid input with JSON errors before launching background work.
- Close analysis clients without allowing cleanup failures to mask the original
  error or a completed result.
- Treat an NVENC timeout as an encoder failure and attempt the bounded CPU
  fallback. CPU timeouts remain terminal and retain diagnostics.
- Reject corrupt/nonfinite/backward word times in cached TTS. Use unique temporary
  filenames when writing audio/cue caches.

Earlier fixes retained and exercised by the suite:

- Validate the source writer's sentence proof objects and repair malformed batches
  without inventing fact/visual IDs. Remove the conflicting prose-only schema.
- Normalize new proof text before creating edit anchors and signatures. Preserve
  valid sealed narration and visual selections during subsequent normalization;
  reject changed signatures instead of silently signing them again.

## Verification

- `python -m unittest discover -s tests`: 539 tests, 5 skipped, no failures.
- Python compilation: 57 application and test files.
- JavaScript syntax checks: `static/js/main.js`, `static/js/ai_video.js`.
- Real FFmpeg tests cover encoding, stream-copy concatenation, decoding,
  subtitles, structured MP4 export and resume without recreating TTS/video stages.
- Replayed task `d7d5eb84-2031-4fc5-864a-30cb87f0159b` from stored data:
  all 37 written scenes retain valid signatures through normalization.
- Fault-injection tests cover interrupted streams, bounded retry, stale tasks,
  startup failure, checkpoint write failure, malformed API input, corrupt TTS
  timing and GPU/CPU timeouts.

Live model quality, service availability, quota and full-length synthesis/export
under current service conditions were not tested. The local suite cannot establish
that every possible external failure has been eliminated.
