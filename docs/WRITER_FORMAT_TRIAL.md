# Reversible flat writer trial

Flat narration-unit prompts now use a direct compact packet instead of inheriting
the legacy paragraph prompt. Each fact text appears once; unit rows contain only
unit IDs and assigned fact IDs. Source ranges and shot bindings remain backend
data, and all existing binding/audit/export checks remain in force. The original
legacy writer and both schema shapes remain available.

On a 503, a multi-event writing group splits between whole situations immediately.
Each child receives preceding completed narration and following event context.
Successful children are saved as canonical candidates before proceeding, so a
later child failure does not discard them. Single situations are not split into
disconnected units; they retain one bounded capacity retry. Other error classes
do not trigger this splitting policy. Writer split IDs are logged explicitly.

Writer and repair prompts explicitly require Vietnamese narration with diacritics,
while retaining confirmed proper names. The existing local language detector now
runs before accepting paragraphs and individual narration units, including cached
drafts. Invalid events alone enter the bounded writer repair; verified source
images and unit IDs stay unchanged. Completed writer caches also require this
language check. This is a heuristic guard, not perfect language identification.

When a video verdict rejects a planned situation, one bounded source-fact review
per scene per run can independently re-read that situation's existing shot pack.
All fact IDs must be returned with confirmed Vietnamese observations; unresolved
reviews remain blocked. The plan and unit focus are updated together without
changing quotas, IDs or shot ranges. The whole affected situation is rewritten
with the preceding and following narration as read-only context, then checked
against both the revised facts and the video. Unrelated situations are retained.
Cache-only final checks never initiate source review. Old observations are kept
as provenance; facts and text change their audit signatures automatically.

Current flat mode uses backend-planned narration units (version 1). The output
rows are `{event_id,unit_id,text}`. The backend assigns unit IDs, source fact IDs
and one or more observed visual IDs before writing; punctuation does not define
the target count. Each situation is still written coherently as a whole. A unit
may contain multiple grammatical sentences or no terminal punctuation. Exact ID
order, nonempty text, selected source coverage, content audit and export checks
remain mandatory. Multiple assigned images are linked within the unit; natural
sentence boundaries are preferred for image changes, with word anchors where
needed. Source shot ranges are never fabricated.

Raw generation responses are saved in `evidence_writing_raw_responses` before
parsing/adaptation (last two attempts per batch). Invalid model rows are therefore
available for diagnosis. Unit boundaries survive cache, normalization, merging
and final output checks; final timeline words must match the recorded unit text.
Old completed section checkpoints are invalidated by alignment version 7; raw
video observations are reusable. `format=legacy` retains the earlier paragraph
protocol and grammatical-sentence checks, and the earlier snapshot remains intact.

`recap_writer_config.json` selects `flat` (current trial) or `legacy`.
`RECAP_WRITER_FORMAT=flat|legacy` overrides that file. Missing/unreadable config
defaults to legacy; unsupported values fail explicitly. No automatic switch to
legacy on 503 is made, so request traces identify which format was tested.

The preceding flat protocol was `{sentences:[{event_id,text,fact_ids}]}`. Its small schema contains
no enums, array bounds or conditional branches. The prompt still asks the writer
to compose each whole situation coherently before returning sentence rows.
Backend checks sentence count, event/fact ownership, source order, narrator-only
style and repeated sentences; proof kind is derived from the cited facts.
Mixed evidence types must be rewritten rather than arbitrarily relabeled.
Canonical paragraphs then use the unchanged binding, signatures and factual
audit. Correct ID references alone do not establish semantic correctness.

Legacy schema/prompt/binding functions remain unchanged in `evidence_writer.py`.
Snapshots of the preceding application, writer and schema-recovery module are
in `backups/writer_before_flat_20261110/`. To roll back behavior, change `format`
to `legacy` and restart the backend; source-video caches are unaffected. Flat and
legacy writing batches use separate keys. Valid canonical drafts can still be
checked and reused. Do not delete `flat_writer.py` while the application imports
it; use the switch, or restore the saved files when removing the trial entirely.

The model, thinking/output limits, 480p/2 FPS visual input, global plan and
sentence targets are unchanged for this experiment. Logs expose `writer_format`
and schema size. Offline tests verify the actual application branch, selective
repair, cached recovery and rollback selection. No live A/B test has been run;
this change does not establish that schema complexity caused the observed 503s.
