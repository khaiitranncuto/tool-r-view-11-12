# Freeze diagnosis: render 09753c89-8c5e-4543-9467-7b6004951d0d

Reviewed output: `C:\video rì view\video_review_09753c89-8c5e-4543-9467-7b6004951d0d.mp4`.
Duration measured using OpenCV: 1126.933 seconds. Persisted render payload selects
`legacy_v1`, with 125 narration rows.

## Confirmed mechanism

The render log at 13:10:10 explicitly reports 30 of 338 visual segments lack a
combined 77.26 seconds of source footage and hold the last frame. In
`video_renderer.py`, `build_visual_edit_plan(...strict_visual_lock=True)` caps
source playback at the selected shot's end. `render_segment` pads the remainder
using `tpad=stop_mode=clone`. This intentionally bakes a frozen image into the MP4
while narration/subtitles continue. It is not evidence of a player/GPU stall.

Frame samples at output 33, 35 and 37 seconds show the same city panorama while
Vietnamese subtitles change. A 5-FPS freeze scan, cropping out bottom subtitles,
also detects approximately:

- 00:32.2–00:37.2: 5.0 seconds.
- 00:43.2–00:47.4: 4.2 seconds.
- 00:53.2–00:58.2: 5.0 seconds.
- 01:04.2–01:11.0: 6.8 seconds.

Those are image-static intervals, which may include static frames already in
the source; the exact manufactured hold budget comes from the renderer's log.
The diagnostic scan was stopped after sufficient evidence was collected; it is
not a certified exhaustive list for the entire output.

## Upstream timing defect

Installed `edge_tts` is 7.3.1. `tts_engine.py` passes `boundary='word'`, whereas
the installed implementation enables word events only for the exact string
`WordBoundary`. Other strings fall through to sentence-boundary configuration.
The locally installed implementation was inspected directly.

The corresponding first cached TTS group has just two cues, each containing a
complete sentence, despite multiple visual beats. Many render-log warnings say
word timing is missing. The beat expansion path therefore estimates cuts using
text weights/partitions. This worsens timing and can allocate too much narration
to an individual six-second source window.

## Root-level corrections needed

1. Request the supported `WordBoundary` mode, validate returned cue granularity,
   and version the TTS cache so old sentence-level cues are not reused as words.
2. Budget the available selected footage against actual spoken duration per
   situation and per cut; a long paragraph must have adequate verified shots.
3. Prevent unnoticed cloned-frame padding. Either produce an explicit diagnostic
   for insufficient footage or revise the selected source group/prose; changing
   transport from TIME/BEAT to JSON alone cannot create missing footage.

This task diagnoses the cause; it does not modify render behavior or re-render
the user's full video. The preserved legacy snapshot remains unchanged.
