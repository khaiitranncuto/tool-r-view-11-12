# Exact sentences (current evidence contract 19)

New situation plans now use the source-grounded writer described in
`SOURCE_WRITER_V19.md`: one existing writing request sees the actual situation
frames and dialogue, and backend code binds the frozen sentences to source IDs.
The older chapter-first/mapping path below remains for legacy projects.

Repair invariant update: targeted repair application, locked-scene coercion and
final narration normalization all enforce the original per-scene sentence quota.
If a repair splits an assigned compound sentence into multiple adjacent factual
sentences, excess boundaries are joined with semicolons. The word sequence must
remain identical and the resulting count must match. Too few sentences are
rejected rather than padded or duplicated. Already-correct prose is unchanged.
This also covers old cached repairs. Replaying c309c4e8 recovers 281 from 286
sentences by changing only punctuation in S0015, S0018 and S0029; no words are
dropped or invented. The final exact-count gate remains in place.

User requirement: minutes -> word budget -> exact spoken-sentence count. A
paragraph, source shot, TIME row or situation is not a sentence. For 20.8 minutes
at the current calibration: 5,620 target words and exactly 281 spoken sentences.
By subsequent user instruction, word counts are reference-only: no chapter,
paragraph or final word-count mismatch blocks completion/rendering. Sentence
tolerance is zero. The reported 727/660-word chapter is allowed when it has its
required 33 sentences. Actual audio duration is
measured separately and is still an estimate before TTS.

1. Full-film overview/evidence establishes the plot and a candidate scene library.
2. Source-ordered situations receive weighted sentence quotas whose sum is exact.
   Important/evidence-rich situations get more sentences. Evidence capacity is
   bounded; shortage raises an error, never silently lowers the target.
3. A text-only chapter request sees plot context, prior narration, grounded facts,
   character permissions and sentence/word quotas. It receives no visual catalog.
   Its JSON has an explicit sentence array for each situation. Sentence counts
   are validated before accepting/caching prose; word counts are measured only.
4. A separate visual request receives the completed prose and candidate images.
   It may choose visual links but must return identical TTS. Changed prose is
   rejected. Scene pools are evidence, not one-shot/one-sentence obligations.
5. Final validation recounts actual spoken sentences, reports words, removes all
   TIME/BEAT/control metadata and checks every paragraph quota. Repair/audit paths
   cannot bypass this check. A deficient script stays failed/partial.
6. The exact contract is persisted early. Before rendering a newly contracted
   project, the request checks the edited script again and requires successful
   analysis. Existing pre-contract projects/legacy rollback remain compatible.
   TTS timing reports real sentences rather than paragraph count. The UI reports
   actual/target sentences and actual spoken words on success.

Completed chapter prose and image mapping have distinct cache entries. Re-running
image mapping does not require re-writing accepted prose. Contract 17 invalidates
older analysis checkpoints. The preserved legacy render snapshot is unchanged.

Offline replay of analysis 03400852 allocated 281 sentences/5,620 words across
95 situations. Its existing 200-sentence, roughly 3,375-word script is rejected
by the new gate. This proves allocation/rejection, not that Gemini has generated
a replacement film script or that the full video has been rendered.
