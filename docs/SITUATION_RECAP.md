# Situation narration and persistent character identities

Evidence contract 16 replaces duration-derived sentence slots in the app's
grounded analysis path. Duration remains an approximate prose budget.

The overview supplies film-wide P IDs, stable visual features, appearance
variants, reference timestamps and name evidence. Every detailed clip receives
that registry plus new identities from earlier clips. Local C IDs remain useful
for observations but cannot rename or replace global IDs. A match requires high
confidence, a stable visual feature plus a second compatible feature from the
profile (including appearance variants), and a shot in the actual clip. Identity
shots need not also be selected as recap shots. Pending matches retain the known
canonical ID, but are not promoted to confirmed links. Truly new identities get
new IDs; repeated known names/neutral aliases resolve to their existing IDs.
Conflicting names remain pending and use neutral prose only in affected events;
names without source evidence are withheld. The registry
is passed through the existing global identity normalizer without another
text-only identity-guessing call.

Detailed extraction groups observations by situation, not by physical action.
`situation_recap.build_situation_plan` orders individual observations by source
time before grouping consecutive runs sharing a situation ID, context and
compatible actors. Intervening storylines split a run: piano -> flooded room ->
hamster stays three runs rather than becoming one crossing source window. One
observed representative source window is selected per fact; all fact proofs stay
in the evidence registry. Full-video, high-confidence `non_story_ranges` exclude
opening/credits/ending montages without applying arbitrary duration cutoffs.
Each situation owns one final prose
paragraph, potentially several sentences and several source shots. The writer
receives the previous eight paragraphs and explicit scene context to avoid
restating decisions/goals and to distinguish returns to earlier storylines. No extra
observations or sentences are requested to fill a duration quota.

The ordinary writer writes final prose directly. It does not draft a paragraph
and then rewrite it into per-shot captions. It may select a representative
subset of the preselected source group, with literal speech anchors for visual
changes. Existing grounding and character checks still apply. Explicit TIME
rows keep the paragraph together through TTS input parsing.

New analyses are required to use the contract; rendering an existing script
does not rewrite it. Offline tests cover identity conflicts, unknown IDs,
paragraph preservation, visual subsets and bypass of the old draft/remap call.
Regression tests reproduce the Shizuka/Gian/Suneo feature mismatch, pending ID
preservation, interleaved scenes and ending-montage exclusion. An offline replay
of the saved 96558824 analysis retained P03=Shizuka, P04=Gian and P05=Suneo through
global normalization and writer permissions. Output fluency still requires
assessment of a newly generated script/video.
