# Source-linked recap (evidence contract 18)

The existing overview, detailed observation, chapter prose and visual mapping
requests are retained. No retry limits or additional Gemini audit passes change.

- Overview non-story ranges are proposals. The existing detailed pass observes
  all frames, including a cold open before the opening song, and supplies actual
  non-story shot IDs. An overview range cannot erase observed story facts.
- Indexed detailed clips now come directly from the original video, keeping
  their global source IDs and playback timestamps independent of proxy sampling.
- Deduplication joins facts only within the same evidence owner and exact shot
  set. A shared or transitively overlapping timestamp cannot merge storylines.
  Conflicting situation claims on one shot are excluded from planning and kept
  in `ambiguous_source_shots`; no owner is invented and no retry is added.
- Each locked visual carries its own fact text, fact ID, evidence owner and
  absolute source bounds. Source descriptions are never taken from another
  event just because it used the same ID. V3 does not rebuild scenes by globally
  redistributing facts after an ordering failure.
- Whisper lines are attached to individual facts by actual source overlap.
  Split situations collect only those lines, and chapter writing receives them
  for causes, decisions and conversation content. Dialogue cannot prove a new
  visible action or identify an unconfirmed speaker.
- Sentence allocation retains the exact final count while accounting for
  selected footage. Visual mapping preserves locked prose and selects actions
  or contextual reactions from that same situation. There is no word-count gate.
- Edge TTS uses the supported `WordBoundary` enum for action timing. Old
  sentence-level timing caches are retained on disk under their old keys;
  new TTS and structured-render cache keys prevent their accidental reuse.

Limitations: this is not a claim that AI recognition is infallible or that legacy
rendering can never hold a frame. Legacy rendering still holds footage when
speech exceeds the available source interval; structured rendering rejects that
shortage. A fresh analysis/export is needed to assess the resulting movie.
Existing exported videos and rollback backups are unchanged.
