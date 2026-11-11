# Evidence-to-sentence workflow — 2026-11-10

The main visual observation pass remains 480p at 2 FPS and LOW media resolution.
The existing lightweight overview is unchanged. No 720p input or higher-FPS pass
has been introduced.

## Flow

1. Overview and detailed observation collect source-linked facts. The detailed
   response has an API JSON schema; every observation identifies SHOT IDs,
   evidence kind and confirmed/uncertain status. Speech remains separate from
   visual claims. Uncertain facts are retained for review, not allocated prose.
2. The backend groups compatible situations and allocates the exact requested
   sentence count to paragraphs. In the active situation workflow, fact count
   and footage length are not upper bounds on the sentence count. Detail,
   dialogue and narrative importance guide the allocation. Every group still
   needs genuine evidence; allocation never authorizes invented or repeated claims.
3. The plan sets a sentence budget per group, without assigning one fixed fact
   or focus to each sentence. All of the group's confirmed facts and dialogue
   remain available. More sentences permit more detail; fewer sentences require
   concise narration of the important developments.
4. The writer receives text only and writes a COMPLETE coherent paragraph for
   each group, using previous narration and upcoming context for continuity.
   It returns `paragraph` and an ordered `sentence_evidence` list of fact IDs and
   evidence kinds. Only after writing does the backend split the paragraph,
   check its exact sentence count and derive edit cues and source references.
   Several distinct sentences may reference different details of the same fact.
   Unknown fact IDs, missing evidence, wrong counts and duplicate sentences are
   rejected. Source image collection, SHOT IDs and source ranges are unchanged.
5. An independent text request checks the written claims against observed facts.
   Contradicted claims go to local content repair. Only an explicit
   `visual_review_required:` verdict opens the corresponding source clip for
   video verification. A text verdict is labeled `text`, never `video`.
6. Verified signatures and verdicts are reused during the final merge. Missing
   verdicts remain pending, and unsupported new prose cannot be silently
   exported through the approximate-image fallback.

## Recovery

Plot coverage is checked before planning: overview milestones retain IDs, source
fact links and explicit uncertain/missing states. New detail observations can
declare `story_beat_ids` only when their visible facts support that milestone,
and `plot_gaps` identify partial/missing links with valid source SHOT IDs.
Legacy time-overlapping observations are candidate sources, never semantic
verification of an entire overview summary. The report deliberately does not
assert all milestones verified, and uncertain overview content cannot become
narration evidence by itself.

Only missing/partial main milestones may request up to two indexed 2 FPS review
windows, at most 60 seconds each, within the existing targeted-review budget.
Reviews use compact known-person context, window-local Whisper and observation
schema; compatible raw responses/supplemental events are cached. Supplemental
events merge into existing chapter checkpoints without adding chapters or
duplicating fact IDs. Review windows with no new evidence remain unresolved;
neither a missing link nor an unconfirmed ending is fabricated. These selective
reviews can consume tokens when needed; the coverage ledger itself is local.

Ending situations receive at most one sentence each when main-story situations
also exist. Climax/resolution remain eligible for detail; unused ending budget
goes to the main story, preserving the exact requested total. No ending is
inferred simply from the last timestamp. Offline replay retains 85 situations
and 281 sentences, with E0084/E0085 receiving one sentence each; no supplementary
request was needed for this cached example. Validation: 618 tests run, 613 passed
and 5 skipped; no live Gemini request was made.

Active writing layout is now `global_batches_v1` (planning workflow 9). The
situation planner creates one global timeline, not three writing sections.
Compatible older plans can be flattened without changing event IDs, evidence,
sentence quotas or source times. Small consecutive groups are generated and
checkpointed, carry previous-tail/next-event context, and concatenate directly
into one response. Final serialization retains one `parts` wrapper for existing
rendering code; it does not repartition scenes or perform another model rewrite.
Three-section response checkpoints are rejected by their mismatched boundaries;
source observation caches remain reusable. Offline replay retained all 85 groups
and 281 sentence slots in 13 writing batches. Tests cover single-timeline source
locking, group resume, direct merging and preserved sealed source signatures.

- Request traces are written to the existing task log under `REQUEST`: prepare,
  send (including local pacing delay), first streamed chunk, success/failure,
  retry wait and stop reason. Each logical call has a local request ID, attempt
  count and request fingerprint. Metadata includes stage/model, content length,
  schema size, output cap, thinking settings, HTTP code, elapsed time and partial
  response size. Prompts, response bodies, credentials and raw error strings are
  not logged by this trace. 503, HTTP 504 and transport timeout are distinguished;
  these observations cannot establish Google's internal capacity/queue state.
  Trace callback failures do not fail generation. No diagnostic API calls are
  added, and the existing bounded retry policy is unchanged.
- Identity cache rebuilds record expected/top-level/chunk versions and retained
  raw-response counts under `CACHE`. The final chapter checkpoint write now
  preserves `character_registry_version`, preventing a correctly versioned chunk
  set from repeatedly invalidating the plan/writing caches on subsequent runs.

- The text writer retries 503 capacity failures twice after the initial request.
  Transport retries and the content-repair round have separate ownership.
- Valid event rows survive malformed neighboring rows and interrupted repairs.
- A new UI project can inherit a failed task's compatible observation, plan,
  completed writing and audit checkpoints. The recovery fingerprint includes
  source path/size/mtime, model, effort, style, prompt, duration, voice rate,
  subtitles, multi-voice mode and evidence contract version. Credentials and
  old task/project/status/error fields are not copied.
- Evidence contract 20 separates new structured observations from old evidence
  caches. Existing files remain intact; older contracts are not represented as
  newly confirmed observations.
- Paragraph response version 2 invalidates only older writing checkpoints;
  compatible observation/image checkpoints remain reusable. The observation
  contract and visual configuration were not changed by the paragraph correction.

## Cost and validation

Strict sentence/source contract: source inventory and total sentence quotas are
checked before generation. Each bound paragraph is checked locally for one proof
per spoken sentence, valid source ranges, exact proof text, fact ownership and
actual selected visual IDs. Before export, the complete target count, immutable
signatures, original video existence, canonical SHOT/time coverage and successful
content audits are required. Approximate visual fallbacks, pending/rejected
audits or missing sentence visuals block export. Reports are saved as
`visual_plan_preflight` and `sentence_visual_contract` and written to the task
log. This verifies structural coverage and recorded audit outcomes; it does not
claim an infallible semantic vision model. The existing selective repair path
handles failed paragraphs; failures never authorize lowering the target count.

Whisper length no longer increases situation sentence-allocation weights:
transcription context is not additional observed visual detail. The main event
and ordered source-backed details remain together in each writer card so prose
is composed as a coherent situation rather than isolated sentences. No diagnostic
or whole-film visual pass is added. Flat/legacy schema selection remains intact.

E0048 correction: fact cards expose chronological `source_order` and evidence
kind. An indirect dialogue sentence must cite an actual dialogue fact with
source speech; nearby Whisper on visual facts does not authorize relabeling.
Visual action proofs retain source-time order. Supported indirect dialogue may
illustrate a nearby reaction in the same situation (at most six seconds after
the cited source window), without replaying earlier action. Local order errors
identify sentence index, fact IDs, previous source time and cited windows for
selective rewriting. Unsupported outcomes remain subject to the factual audit.
Offline replay rejects E0048 sentence 4's wrong evidence label; a four-sentence
source-backed rewrite binds V001, V001, V002, V003 without changing observations.
Validation: 603 tests, 598 passed, 5 skipped; no live Gemini request.

Narrator-only policy: Whisper is supplied in `transcript_context_only`, separate
from visual fact cards. It supports plot comprehension; the writer must use
indirect narrator prose, ignore incoherent ASR and never copy dialogue verbatim.
Quoted speech/dialogue turns are rejected locally for selective paragraph repair
and checked again before audit/export. Completed cached parts containing dialogue
are resumed only up to the clean prefix; video observations remain reusable.
This does not remove transcript source records or authorize visual claims from
speech. Clear dialogue meaning may support indirect narration, never a guessed
speaker or visible action.

Sentence counting now shares a quotation-aware splitter with paragraph binding
and repair. In the stored E0078 response, interior quoted questions/exclamations
no longer inflate 10 narrator sentences to 12. The old response is still rejected
for direct dialogue rather than passed by merely stripping quotation marks.
Local validation: 600 tests, 595 passed, 5 skipped; no live Gemini call.

Paragraph evidence-count correction: the provider receives one shared object
shape with batch IDs, fact IDs and shared array bounds. Exact per-event quotas
and fact ownership are enforced locally, without repeating an `anyOf` branch per
event. This replaces the branching schema associated with the 400 failure in
task `1affc590-598d-43c8-ab88-25554f542099`; its first-batch schema decreases from
5,445 to 1,085 serialized bytes. Google documents that complex schemas can be
rejected: https://ai.google.dev/gemini-api/docs/structured-output.

Writer requests rejected with schema-related/generic INVALID_ARGUMENT 400 can
retry once without a response schema, retaining JSON MIME type, model, prompt,
generation options and all backend validators. Other status codes and explicit
invalid-key errors do not trigger this recovery. A second failure propagates.
No extra request is made on success. Prompt instructions distinguish one proof per sentence from one
proof per fact/evidence kind. Count errors report expected and actual lengths;
selective repair retains valid neighboring paragraphs. The E0062 stored failure
contained one sentence and two proof entries (visual plus dialogue). Neither
proof is silently discarded or relabeled. Paragraph binding also passes the same
normalized source-time catalog to the lower-level binder, including resumed
sources that store only numeric seconds. Local suite: 597 tests, 592 passed,
5 skipped. Tests include actual SDK streamed requests through an offline HTTP
transport simulating 400 followed by JSON recovery; no live model request was made.

For S0007, the existing Gemini name Nobita can be recognized as a dialogue
mention despite the unique one-letter ASR typo "nôpita". Short/ambiguous typo
matches are excluded. This changes allowed/mentioned names only, never speaker
or visible-person identity, and leaves the original transcript unchanged.

The writer does not upload or send video. The paragraph correction was replayed
against failed task `311c7bd7-1d2d-4125-88f0-58cb9521730e`: its original 85 groups
and 250 facts now receive exactly 281 sentences (93, 73 and 115 across the three
parts). All facts and visual-source records remain unchanged. This verifies the
allocation regression, not the quality of an unrequested live model response.

These are offline structural measurements, not measured Gemini token savings.
The original V2 archive did not record API usage, so equal-or-lower total tokens
relative to that archive cannot yet be asserted. Initial visual observation,
text verification, retries and any selected video reviews still consume tokens.

Tests cover the actual application writer/audit functions, exact sentence counts,
schema/slot errors, selective repair, unchanged signature/cue/timeline/render
plans through merging, selective video verification, cache reuse and HTTP retry
into a new project. No live Gemini/Groq/Edge TTS requests are part of these tests.

Final local validation after the paragraph correction: 563 tests run, 558 passed
and 5 skipped. Paragraph tests include compound facts supporting several distinct
sentences, all evidence reaching the writer even with a small quota, exact counts,
punctuation-preserving splitting, source immutability, selective repair, cached
paragraph recovery and sealed proof preservation through merging.

## Persistent character identity

Updated policy (registry version 3): Gemini is the primary naming source.
High-confidence Gemini overview names are retained without Whisper approval;
Whisper remains supplementary dialogue evidence. `name_source=gemini` records
this decision without falsely marking names as transcript-verified. Unique
existing names resolve colliding clip-local IDs; conflicting people cannot add
name candidates to or inherit the colliding person's display name. Version 2
normalized checkpoints rebuild from stored raw responses under this policy.

Speech-content observations such as "explains the history of migration" without
linked spoken text are excluded from situation planning and writer evidence,
even when a legacy observer mislabeled them as visual. Original facts remain
stored. Visual actions in the same situation remain eligible, and sentence
quotas remain controlled by the sentence contract. The writer is instructed
to select dialogue evidence only where actual dialogue context is available.

Character registry version 2 keeps append-only person IDs across observation
chunks. Identity certainty and name certainty are separate. A confirmed name
requires structured dialogue/on-screen evidence linked to the relevant source
time; third-person mentions and generic "dialogue" labels are insufficient.
Unconfirmed people retain stable descriptive aliases. Face/body descriptors,
distinctive traits and clothing variants help matching; ambiguous matches remain
pending rather than overwriting another person's identity.

New observations use explicit `[CHAR=id]` references. Rendering those references
does not replace arbitrary words in facts. Original observation text is retained.
This fixes the legacy Gian substitution that corrupted "thời gian" and caused
the S0028 narration rejection. Older normalized identity/writing checkpoints are
rebuilt from compatible stored raw observations, preserving the visual cache.

At most two optional reference frames per person are cached locally from existing
2 FPS clips. These files are not uploaded in extra model calls. Requests receive
a compact text registry without local image paths or appearance history. Main
480p/2 FPS input and the existing overview sampling remain unchanged. This adds
no API pass; identical total token cost and perfect identity recognition are not
guaranteed.

Local validation after the identity correction: 582 tests run, 577 passed and
5 skipped. Offline replay of all 13 stored observation responses from task
`309060ab-af6f-4ab6-8ee8-b839efa3d15d` preserved the original time-machine fact
through both identity stages and retained the exact 281-sentence allocation.
No live Gemini/Groq/Edge request was used for this validation.
