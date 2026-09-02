# Nephrolex retrieval ablation

Gold set today: 67 answerable questions with pinned chunk-level relevance, plus scope and adversarial cases scored separately. The historical rows below were measured against a 50-question set. `dev` is tuned on; `test` is held out.

**Historical, and not comparable to the current system.** Two things changed after these runs: scoring was corrected to treat KDIGO's two printings of a recommendation as one gold item, and the gold set grew from 51 to 67 answerable questions with harder paraphrase coverage. Neither can be applied backwards - these runs stored only their top 10, and their chunk IDs resolve against corpora since rebuilt (0% of the original baseline's IDs exist today), so a re-score would find no duplicates and report a meaningless "no change". For the shipped configuration under both metrics see the table below and the honesty notes in EVALUATION.md.

| Pipeline stage | nDCG@10 | recall@10 | recall@20 | MRR@10 | P@5 | dev | test |
|---|---|---|---|---|---|---|---|
| Original: RRF + hand-written boost ladder | 0.2760 | 0.3967 | 0.4900 | 0.2549 | 0.0960 | 0.3427 | 0.1341 |
| Normalised weighted fusion, boosts removed | 0.4919 | 0.5933 | 0.6433 | 0.4964 | 0.1440 | 0.5123 | 0.4486 |
| + index section paths & generated questions | 0.5154 | 0.6100 | 0.6467 | 0.5253 | 0.1600 | 0.5330 | 0.4779 |
| Corpus rebuilt from Docling (layout-aware) | 0.4466 | 0.5627 | 0.6477 | 0.4219 | 0.1569 | 0.4735 | 0.3876 |
| + query/expansion blending, length damping | 0.5017 | 0.5954 | 0.6935 | 0.4787 | 0.1608 | 0.5447 | 0.4078 |
| + table context, mirror repair, range matching | 0.5235 | 0.6020 | 0.7359 | 0.5085 | 0.1725 | 0.5763 | 0.4078 |
| + dense: bge-small-en-v1.5 (33M) | 0.5914 | 0.6706 | 0.7268 | 0.5788 | 0.2118 | 0.6520 | 0.4588 |
| + dense: bge-m3 (568M) | 0.5983 | 0.6935 | 0.7637 | 0.5757 | 0.2196 | 0.6514 | 0.4821 |
| + cross-encoder rerank: bge-reranker-v2-m3 | 0.6705 | 0.7457 | 0.7980 | 0.6751 | 0.2471 | 0.7052 | 0.5949 |
| swap dense -> MedEmbed-large (medical domain) + rerank | 0.6819 | 0.7572 | 0.8013 | 0.6890 | 0.2471 | 0.7145 | 0.6106 |
| rerank depth 50 -> 30 (faster and better) | 0.6873 | 0.7768 | 0.7980 | 0.6876 | 0.2510 | 0.7244 | 0.6061 |
| drop 84 front-matter / appendix chunks | 0.6898 | 0.7768 | 0.8046 | 0.6911 | 0.2549 | 0.7255 | 0.6117 |
| per-row table bboxes; drop abbreviation glossary | 0.6880 | 0.7768 | 0.7980 | 0.6870 | 0.2510 | 0.7231 | 0.6111 |
| fusion weights 0.30/0.20/0.30/0.20 | 0.6930 | 0.7719 | 0.8046 | 0.7059 | 0.2549 | 0.7302 | 0.6117 |
| restrict corpus to guidance sections | 0.7014 | 0.7752 | 0.8013 | 0.7161 | 0.2588 | 0.7404 | 0.6161 |
| figure captions as retrievable chunks | 0.6969 | 0.7703 | 0.7964 | 0.7183 | 0.2510 | 0.7338 | 0.6161 |
| drop BM25 from scoring (kept for candidate recall) | 0.7270 | 0.8183 | 0.8379 | 0.7304 | 0.2627 | 0.7561 | 0.6634 |
| weights 0.30/0.40/0.30, honest to the noise floor | 0.7280 | 0.8275 | 0.8428 | 0.7302 | 0.2588 | 0.7523 | 0.6751 |

## Shipped configuration, both metrics

| Scoring | nDCG@10 | recall@10 | recall@20 | MRR@10 | P@5 |
|---|---|---|---|---|---|
| as built | 0.6927 | 0.7766 | 0.8132 | 0.6943 | 0.2388 |
| corrected | 0.7060 | 0.8010 | 0.8371 | 0.6943 | 0.1970 |

P@5 *falls* under the correction, because a duplicate in the top 5 no longer counts as a second relevant result. A correction that costs something on one metric is a correction rather than an adjustment.

### Every figure above is on the pre-re-annotation answer key

The gold set was re-annotated by concept rule on 2026-08-19 (`scripts/reannotate_gold.py`,
27 judgements across 8 cases, rationale in `reports/EVALUATION.md`). The tables above are
**not** re-scored under it and cannot honestly be: the earlier stages stored only their
top 10, so recall@20 is unrecoverable, and their chunk IDs resolve against corpora since
rebuilt. Re-scoring them would silently find nothing and report a confident no-change for
a comparison never made — the same defect this project has now caught three times.

They stay as measured, labelled. The comparable current figures on the new key are
nDCG@10 0.7124, recall@10 0.8014, recall@20 0.8346, MRR@10 0.7121, and every ablation
delta above was measured with baseline and variant on the same key as each other, which
is what a delta requires.

## Numeric bands: two silent failures and a prior that did not mean what it said

Chasing the staging questions that survived the corpus repairs turned up three defects
in a row, each hidden by the one in front of it.

**The comparator glyph.** KDIGO's PDF maps the "greater than or equal" character to a
dollar sign, so thresholds were stored as "urine ACR $ 30 mg/g" and "eGFR $ 60 ml/min" -
69 occurrences against 71 where the real glyph survived, the same threshold written both
ways in the same corpus. That is wrong text on a citable chunk before it is anything
else. Repaired in `normalize_text`; currency was ruled out first (every occurrence has a
space before the digit, none is adjacent, no cost word appears near one).

**The bound pattern.** `BOUND_RE` listed `>=` and `<=` but not the Unicode comparators
the guidelines actually use, so GFR categories G1 (">= 90") and G5 ("< 15") - precisely
the two a range cannot express - had no band at all. "Their filtration number is 95, is
that in the normal band?" scored 0.000 against the row that answers it.

**The prior.** With both fixed the signal was correct and still changed nothing, because
`NUMERIC_BAND_PRIOR` was 0.10: a perfect band match moved a chunk about 7%, halved again
by the rerank blend. The file's own docstring called this "a hard match, not a soft
similarity"; the constant did not act on it. The cross-encoder cannot substitute, because
it is being asked to choose between table rows that differ only in their numbers.

Swept on the gold set, its staging slice and the held-out set:

| prior | gold set | staging | held-out | clinician |
|---|---|---|---|---|
| 0.10 | 0.7091 | 0.4629 | 0.6941 | 0.5305 |
| 0.25 | 0.7069 | 0.4894 | 0.7042 | 0.5484 |
| **0.40** | **0.7152** | **0.5546** | 0.7047 | 0.5492 |
| 0.60 | 0.6851 | 0.5469 | 0.7054 | 0.5504 |

0.40 ships. At 0.60 the prior overrides relevance instead of nudging it and the gold set
falls away, which is the behaviour the "deliberately mild" comment about priors predicts.

Together with the corpus repairs: gold set 0.7124 -> 0.7152, recall@20 0.8346 -> 0.8483,
paraphrase slice 0.4133 -> 0.4546, end-to-end audit 67/75 -> 69/75 clean.

## Kept despite a measured cost: table row labels

Docling emits a cross-tab with an empty corner cell, and `parse_tables` built each row
as `header: value` pairs while skipping any cell whose header was blank. The blank
header is the corner, so the cell it discarded was the one naming the row. NICE's
monitoring-frequency table came out as six rows of bare counts - "A1: 1; A2: 1; A3: 2" -
with nothing saying which GFR category each row was for. Verified against the source PDF:
the numbers were right and unattributable. 245 label cells across 25 tables were lost
this way.

Fixing it alone cost 0.0075 nDCG@10 with an interval excluding zero, and chasing that
loss found a second defect. Rows of three tables of *laboratory value variation by age
group* had been too short to clear the indexing threshold precisely because their labels
were being dropped; restored, they outranked the real GFR staging table. The reason was
not the numbers. Docling repeats a spanning header across every column it covers, so
those rows each carried "GFR category (ml/min per 1.73 m2)" seven times over - keyword
stuffing produced by the parser, not by the guideline. Emitting each distinct header once
recovered it.

Both fixes together: nDCG@10 0.7124 -> 0.7091, 95% CI [-0.0101, +0.0018], within noise;
recall@20 0.8346 -> 0.8334; held-out 0.7014 -> 0.7029 with the clinician slice unchanged
at 0.5461. So 245 restored row labels and the removal of a keyword-stuffing artefact cost
nothing measurable.

What remains on those staging questions is a row-selection problem rather than a corpus
one: for "which GFR category covers 60 to 89", the right table now ranks first and the
wrong row of it ranks above the right row. Reverting the corpus is `data/chunks_v2.bak`.

## A sibling that completes a section is not a sibling that repeats it

`diversify()` damps a second chunk from the same section by 0.80 per repeat, to stop
five near-identical chunks filling the results. It could not tell a duplicate sibling
from a complementary one, and the difference turned out to matter clinically.

Asked what blood pressure target applies in CKD, the model produced the correct and
complete NICE answer - below 140 mmHg under ACR 70, below 130 mmHg at or above it - and
the verifier discarded it on three counts: "130 mmHg", "80 mmHg" and the range "120 to
129" were not in the cited evidence. They were not, because NICE 1.6.2 states all three
and had been pushed to rank 8, outside the citable window. The model had read them from
the neighbouring page text, which the prompt marks "for understanding only, do not cite"
and which `verify()` deliberately refuses to count as support. Clinically right,
evidentially wrong, and caught.

The demotion was not a scoring failure. 1.6.2 scored 0.816, higher than the three chunks
ranked above it; the section penalty took it to 0.653 because 1.6.1 had already been
selected from "Blood pressure control". But 1.6.1 and 1.6.2 are the two arms of one
decision, not two ways of saying the same thing - and NICE 1.6.4, a pointer to another
guideline carrying no number at all, sat unpenalised at rank 2 because it lives in a
different subsection.

A chunk is now exempt from the penalty when it states a threshold none of its selected
siblings state. 1.6.2 offers 130 mmHg, 80 mmHg and 129 mmHg, none of them in 1.6.1, so
it keeps its score and returns to rank 4.

| | off | on |
|---|---|---|
| 67-question gold set | 0.7061 | 0.7075, CI [-0.0062, +0.0103] |
| held-out chunk-first | 0.8440 | 0.8411, CI [-0.0085, +0.0000] |
| clinician-FAQ success@5 | 20/20 | 20/20 |
| clinician-FAQ nDCG | 0.7008 | 0.7225, CI [+0.0000, +0.0500] |
| correct-row precision | 18/19 | 18/19 |
| gold-set recall@20 | 0.8250 | 0.8312 |

Every interval spans zero: no measurable gain, and no measurable cost. It is kept for
the mechanism rather than the number - it removes a way for a correct answer to be
unciteable, and the aggregate metrics never saw the problem in the first place.

The risk worth naming: almost any two chunks differ by *some* number, so an exemption
this shape could let genuinely redundant siblings through. Correct-row precision is the
canary - sibling table rows differ only in their numbers - and it is unchanged at 18/19.
`COMPLEMENTARY_EXEMPTION = False` reproduces the "off" column.

`SHOWN_EVIDENCE` was also raised 6 -> 8 while diagnosing this, which put 1.6.2 in reach
before the ranking was fixed. Both changes stand; either alone would have sufficed for
this question, and only the ranking fix addresses the mechanism.

## Three constants, re-swept on the current pipeline

Every one of these had been chosen against a pipeline that no longer exists - before
doc2query, before the reranker read the expansions, before the corpus repairs, before
the numeric-band prior moved. Stale rather than wrong, and worth re-testing rather than
inheriting. All three came back at the value already shipped.

| knob | shipped | swept | outcome |
|---|---|---|---|
| `w_doc2query` | 0.29 | 0 / 0.21 / 0.29 | 0.21 and 0.29 indistinguishable; 0.29 is also the value that makes the weights sum to 1 |
| `CANDIDATE_POOL` | 120 | 120 / 240 | doubling gains nothing; both intervals span zero |
| `rerank_depth` | 30 | 20 / 30 / 50 | 30 best on every metric; 50 *loses* a FAQ question |

**doc2query weight.** Off costs FAQ success@5 15/20 and held-out 0.6143, so the signal is
not marginal. At 0.21 and 0.29 both sets reach 20/20 and both clear their interval
against zero, while differing from each other by 0.011 (main) and 0.017 (held-out)
against interval widths of +/-0.06 to 0.09 - indistinguishable. 0.29 is kept because it
is nominally ahead on both and because 0.21/0.29/0.21/0.29 sums to exactly 1.

**Candidate pool.** Reachability analysis over the 244 grade-2 gold chunks in all three
answer keys: top 30 reaches 91.8%, top 120 96.7%, top 240 99.2%, top 500 100%. So 120
leaves 8 chunks structurally unreachable - and doubling to 240 converts none of it, +0.0038
main (CI [-0.002, +0.012]) and -0.0014 held-out. Those chunks enter the pool around rank
150-400 and never reach the rerank window. Reachable is not retrieved.

**Rerank depth.** 30 beats both neighbours on main, held-out and FAQ. Depth 50 drops the
FAQ set from 20/20 to 19/20: a deeper window admits competitors the cross-encoder prefers
over the right answer, the same mechanism that buried `mon_acr_paraphrase` before the
reranker was given the generated questions to read. The original "30 beats 50" finding
was made on a much earlier pipeline and replicates here, which is worth more than a
number that was never re-questioned.

**On the timings in those runs.** The sweeps memoise cross-encoder scores across
settings - exact, since a (query, chunk) score does not depend on the weight or the
depth - which cut the doc2query sweep from 224s to 14s per setting at 31% cache hits.
It also makes the per-row `seconds` and `ms/query` columns meaningless for comparison:
later rows reuse earlier work, so they measure the cache. Those columns show progress,
not cost, and the scripts now say so. Latency needs its own uncached pass, which was not
required here because the winning depth is also the cheaper one.

Each run records its own provenance - corpus size, dense vector shape, index file sizes
and modification times, the fusion weights and priors in force - so two runs that
disagree can be told apart. That exists because a report in this project once recorded
`dense_model: MedEmbed` while running lexical-only, 0.064 nDCG adrift, with nothing
capturing whether the vectors had loaded.

## Measured and kept: clinician-voice document expansion

Four techniques were measured and rejected before this one, all of them attacking the
query. This attacks the documents, and it is the only one that cleared its confidence
interval.

The held-out chunk-first set showed retrieval swinging from nDCG 0.90 to 0.38 on
identical content depending only on whether the question used guideline vocabulary. The
per-case audit named the mechanism: a clinician asks about "a flozin", a word in neither
guideline; about a "vomiting bug" where KDIGO writes "acute, dehydrating illness". An
LLM writes, for each chunk, the questions a clinician would actually ask that it
answers - 4172 of them over all 1479 chunks, generated once and cached, costing nothing at
query time.

| | clinician-phrased | echo-phrased | main gold set |
|---|---|---|---|
| without | 0.3849 | 0.9000 | 0.7125 |
| with (w=0.4) | **0.5461** | 0.9033 | 0.7124 |
| delta | **+0.1612** | +0.0033 | -0.0001 |
| 95% CI | **[+0.0747, +0.2581]** | [-0.0803, +0.0730] | [-0.0412, +0.0311] |

It lifts the weak side without costing the strong one, and does not move the
question-first gold set outside its noise floor while cutting total misses there from 5
to 3. The phrasing gap falls from 0.515 to 0.357.

Two design choices carry the result:

**Indexed separately, never concatenated into the chunk.** Doc2Query++ reports that
concatenating expansions harms dense retrieval by dragging the passage embedding away
from what the passage says. The stronger reason here is structural: generated text lives
in its own index and its own signal, so it can influence ranking and can never reach a
citation, a quote or the verifier. A hallucinated question costs ranking and nothing
else. That is a safety property, not an optimisation.

**Not the corpus's existing `hype_questions`.** Those are template-built from topic
labels - 560 distinct strings across 3106 instances, one repeated 572 times - and
`evaluate_gold.py` already excludes topic labels from scoring as circular. Checked
rather than assumed over the generated set: 0 of 4172 match a template question, 0 are
built only from topic-label words, and 68% of the words in a generated question do not
appear in its source chunk.

Full sweep, per-case mechanism and the pre-registered prediction this confirmed are in
`reports/doc2query_interim.md`.

## Correct-row precision, and the spelled-number gap it exposed

Aggregate metrics treat a near miss as a small loss. Returning the *wrong row of the
right table* is not a small loss: it is a wrong answer carrying a correct-looking
citation to a real KDIGO table, and nDCG cannot see it. So it is measured directly, by
`scripts/check_table_rows.py`, over every gold question in all three answer keys whose
answer is a specific table row.

**18/19 = 95% correct-row precision.** The table surfaces but the wrong row leads once.

Finding it found a defect. Asked "how much salt a day is right for a child of eleven?",
the system ranked KDIGO Table 22's *0-6 months* row - 0.110 g/day - above the 9-to-13
year row that answers it at 1.2 g/day. A tenfold paediatric dosing error, with
provenance, scoring as ordinary partial credit everywhere else.

The cause was that `numeric_range_signal` extracts digits, and "eleven" is a word. With
no number to anchor on the band signal cannot fire, and *nothing to anchor on is not a
neutral outcome* - it hands the choice to a cross-encoder being asked to separate rows
that share a header, a section, a caption and a sentence structure, and differ only in
their numbers. Substituting "11" scores the correct row 1.0 and the infant row 0.0.

`spelled_numbers` now parses number words, so "a child of eleven" and "a child of 11"
behave identically, and "sixty-five" reads as 65. Word boundaries are load-bearing:
without them "one" fires inside "money" and "ten" inside "often", feeding junk into a
signal meant to be a hard match.

| | before | after |
|---|---|---|
| correct-row precision | 17/19 = 89% | **18/19 = 95%** |
| held-out nDCG@10 | 0.8407 | **0.8516** |
| held-out recall@10 | 0.9130 | **0.9420** |
| clinician-FAQ success@5 | 20/20 | 20/20 |
| 67-question gold set | 0.7080 | 0.7074 (-0.0006, noise) |

The one remaining wrong-row case is `stag_g2_paraphrase` ("filtration came back at 72
with protein in the urine"), which returns an albuminuria row where a GFR row is wanted -
visibly wrong to a reader, unlike the paediatric case.

## Shipped after being reverted: doc2query-aware reranking

The cross-encoder is half the final score and scores `(query, raw_text)` pairs, so it
cannot see the generated questions that surfaced a candidate. Asked "how much does ACR
have to rise before it means something?", the doc2query signal ranks KDIGO Practice
Point 2.1.5 first on its own; the reranker then fills the top four positions with
near-identical rows of NICE's monitoring table and pushes the answer past rank 20.
Appending the expansions to what the reranker reads fixes it.

This was built, measured, reverted, and reinstated. The reversal was correct on the
evidence available then; a third benchmark changed it.

| benchmark | off | on |
|---|---|---|
| clinician-FAQ, success@5 (20 q, externally sourced) | 90% | **100%** |
| clinician-FAQ, nDCG@10 | 0.5456 | **0.7013** |
| held-out chunk-first (46 q) | 0.7047 | **0.8407** (+0.1360, CI [+0.0538, +0.2299]) |
| held-out, clinician-phrased slice (26 q) | 0.5492 | **0.7625** |
| 67-question gold set, nDCG@10 | 0.7152 | 0.7080 (-0.0072, within noise) |
| 67-question gold set, recall@20 | **0.8483** | 0.8184 |
| 67-question gold set, total misses | **2** | 5 |
| end-to-end audit | **69/75** | 67/75 |

The first rejection reasoned that the only benchmark favouring it - the chunk-first
held-out set - shared an origin with the feature, since both its questions and the
expansions are LLM-written from the same chunks. That objection does not apply to the
FAQ set, whose questions come from the primary-care literature on what clinicians ask.

What the extra misses actually are: three of the four newly-failing gold questions were
already scoring 0.185 to 0.316 and tipped below zero, against one genuine 1.000 -> 0
(`mon_aki_paraphrase`) and one 0 -> 1.000 rescue (`mon_acr_paraphrase`). That is why the
nDCG delta is noise.

The caveat kept with it: `egfr_confirm` and `faq_04` are the same clinical question in
different words, and this fixes one while breaking the other. It trades phrasing
sensitivity rather than removing it. Both configurations remain measurable -
`rerank_with_doc2query=False` reproduces every figure in the "off" column.

## The premise gate was reading evidence it could never cite

The gate refuses a question built on a threshold neither guideline states. It worked by
pooling the raw text of the top 20 retrieved chunks and asking whether the queried value
appeared anywhere in them, either exactly or inside a closed band with the same unit.
Open-ended bounds are deliberately excluded, which is what stops `>= 90 ml/min` vouching
for any large number.

Support could therefore arrive from a chunk in an unrelated clinical context. Asked *at
what eGFR of 5 ml/min an SGLT2 inhibitor should be started*, the premise was accepted:

| | |
|---|---|
| closed bands found, unit ml/min | (20, 45), (20, 45), (20, 45), **(5, 15)** |
| source of the band containing 5 | KDIGO Practice Point 5.4.5 |
| what that chunk is about | the eGFR window for pre-emptive transplantation **in children** |
| its rank | **8** |

The number is real, stated, and in the right unit. It has nothing to do with SGLT2
initiation. The same mechanism accepted *"why do the guidelines recommend an eGFR target
of 75 ml/min"*, where 75 falls inside the stated ranges 60-89 and 42-78.

**The obvious fix does not work.** Restricting the gate to the citable window,
`quotable[:SHOWN_EVIDENCE]`, changes nothing: the paediatric chunk is ranked 8th, so it
is inside that window. Measured across all 75 gold cases plus three probes, 0 of 78 case
statuses moved.

What does work is reading a *shallower* window than the answer may cite. The
justification is not that shallower is safer in general - it is that a premise is a
stronger claim than a citation. It asserts the value is a real threshold **for this
question**, not merely that the corpus mentions it somewhere.

| depth | gate fires | new catches | false declines |
|---|---|---|---|
| 3 | 5 | 2 | 1 - `stag_g2_direct` |
| 4 | 5 | 2 | 1 - `stag_g2_direct` |
| **5** | **4** | **2** | **0** |
| 6 | 3 | 1 | 0 |
| 8 | 2 | 0 | 0 |
| pooled top 20 | 2 | 0 | 0 |

`PREMISE_EVIDENCE_DEPTH = 5` ships. End-to-end audit at that setting: **69/75 clean, 0
unsafe answers, 0 false declines**, status distribution unchanged at 67 answered / 6
declined / 1 out-of-domain / 1 premise-not-found. Every remaining failure is a retrieval
miss, none is a safety failure.

### What it fixes, and what it does not

Tested before claiming, because the previous version of this note over-claimed:

| probe | before | after |
|---|---|---|
| eGFR **of 5 ml/min** for SGLT2 - support rank 8, paediatric | answered | **premise_not_found** |
| eGFR **target of 75 ml/min** - support in 60-89, 42-78 | answered | **premise_not_found** |
| SGLT2 threshold **"set at 45 ml/min"** - 45 is in Rec 3.7.3, rank 1 | answered | answered |
| ACR **"above 30 mg/mmol before referral"** - 30 top-ranked, in context | answered | answered |
| *At what eGFR should an SGLT2 inhibitor be started?* (control) | answered | answered |

So the gate catches **contamination**, not **false relationships**. Where the number is
genuinely stated in exactly the context asked about, a wrong claim *about* that number
is still answered - and the verifier is what stands behind it, since a generated
sentence must trace every value to its own citations.

The caveat kept with it: the boundary is one case wide on each side. Choosing 5 over 4
rests on a single false decline, and 5 over 6 on a single probe - and all three probes
were written by this system's author, the same circularity the gold set carries. What is
not thin is the protective number, zero false declines across all 75 gold cases, and the
fact that tightening fails conservatively - a refusal listing the closest material,
never a wrong answer. `PREMISE_EVIDENCE_DEPTH = None` reproduces the pooled behaviour.


## Measured and rejected

### MedCPT (a query encoder trained on real user queries)

The held-out set showed retrieval swinging from nDCG 0.90 to 0.38 on identical content
depending only on whether the question used guideline vocabulary. MedCPT is the
literature's most direct answer to that: an asymmetric pair of encoders trained
contrastively on 255 million PubMed click logs, so the query side has seen how people
actually type. Wired in `nephrolex/retrieval/medcpt.py` - two checkpoints, [CLS]
pooling, 64/512 token limits, articles fed as [title, body] pairs. Loading it through
SentenceTransformer would have mean-pooled a [CLS] model and measured the harness.

Dense-only, so nothing else dilutes the encoder:

| encoder | clinician-phrased | echo-phrased | gap |
|---|---|---|---|
| MedEmbed-large-v0.1 | 0.3660 | 0.7518 | 0.386 |
| MedCPT | 0.3059 | 0.4852 | **0.179** |

MedCPT halves the phrasing gap and is rejected anyway, because it halves it by being
uniformly worse rather than by helping the phrasing it was chosen for: clinician -0.060
(95% CI [-0.215, +0.102], noise), echo -0.267 (95% CI [-0.402, -0.136], significant).
Reporting the gap alone would have made this look like a success. In the full pipeline
it is 0.3512 / 0.8779 against MedEmbed's 0.3849 / 0.9000 - slightly worse on both.

The likely reason is genre: MedCPT's training pairs are queries against PubMed titles
and abstracts, and a guideline corpus of table rows and numbered recommendation atoms
is a different kind of text from an abstract.

Two things this measured that matter more than the swap:

**The failure set does not move.** The same eight held-out questions are total misses
under both encoders, dense-only and reranked. They fail for structural reasons - a
colloquialism that appears nowhere in either guideline ("a flozin"), a figure caption
retrieved instead of the figure body that holds the thresholds, a question about the
*difference* between two measures where the corpus states only each measure. No
embedding space fixes those, so the encoder branch is closed and the remaining
candidate is document-side expansion.

**The reranker absorbs encoder quality.** Dense-only the two encoders differ by 0.267
nDCG on echo phrasing; with `w_rerank=1.0` the cross-encoder re-orders the top 30 and
that difference collapses to about 0.02. The earlier finding that seven embedding
models span only 0.586-0.604 is correct, but the reason is not that the models are
alike - it is that this pipeline's reranker makes them alike. That is a claim about the
architecture rather than about embeddings.

### HyDE (query rewriting into guideline register)

The paraphrase slice fails because question and corpus use different words. Contextual
Retrieval enriched the documents and ColBERT changed the matching function; neither
touched the query, which is where the mismatch is. HyDE does: an LLM writes the passage
a guideline would contain if it answered the question, and that is embedded as the
search key. Nothing from it reaches the answer, the citation or the verifier, so a wrong
hypothetical costs ranking and never grounding.

Two formulations were measured on the re-annotated key, all 67 answerable questions,
hypotheticals cached so a run is reproducible and free (`reports/hyde_sweep.json`):

| variant | nDCG@10 | Δ | 95% CI | recall@20 | paraphrase Δ |
|---|---|---|---|---|---|
| baseline | 0.7165 | — | — | 0.8296 | — |
| max | 0.7213 | +0.0070 | [−0.0134, +0.0268] | 0.8298 | +0.0266 |
| blend 0.7 | 0.7219 | +0.0076 | [−0.0128, +0.0285] | 0.8284 | +0.0416 |
| blend 0.5 | 0.7163 | +0.0020 | [−0.0164, +0.0207] | 0.8188 | +0.0247 |

`blend` is the published formulation and moves the single query vector toward the
hypothetical, which is destructive: a correct hypothetical written in KDIGO's register
pushed the NICE gold chunk for `def_paraphrase` out of the top 20 entirely. `max` keeps
both vectors and takes the better match per chunk. It has no weight to fit, which is why
it is reported first despite blend 0.7 edging it on the mean.

Rejected as a default: no variant clears its own confidence interval overall, and it puts
an LLM call on the query path. The paraphrase-slice effect is the interesting part — max
gives +0.0266 with CI [−0.0025, +0.0613], a lower bound sitting on zero — so this is
"not demonstrated on 15 questions", not "does not work". Kept behind `hyde_mode`, off.

One correction worth recording: raising a chunk's score cannot lower its rank was the
reasoning for expecting `max` to be safe, and it is false. For `gfr_g5` the hypothetical
lifted 66 other chunks past the gold one. For `def_paraphrase` `max` *improved* the gold
chunk's dense rank from 18 to 3 and the final answer still lost it — the cross-encoder
scores an unchanged (query, chunk) pair, so what changed was which competitors reached
its window. Widening recall hands the reranker new candidates it may prefer.


Techniques tried against the shipped configuration and dropped. They are listed because a negative result with a mechanism is a stronger answer to "why didn't you use X?" than never having measured it. Every row here, baseline included, was measured on the 51-question gold set these experiments ran against - like for like. The current system scores differently on the larger set; see EVALUATION.md for that number.

| Variant | nDCG@10 | recall@10 | recall@20 | MRR@10 | P@5 | paraphrase |
|---|---|---|---|---|---|---|
| **baseline: shipped config, same gold set as these rows** | **0.7280** | **0.8275** | **0.8428** | **0.7302** | **0.2588** | **0.4455** |
| Contextual Retrieval, full generated context | 0.7021 | 0.7980 | 0.8324 | 0.7082 | 0.2588 | 0.3491 |
| Contextual Retrieval, generic words stripped | 0.7051 | 0.8029 | 0.8248 | 0.7045 | 0.2549 | 0.3491 |
| Contextual Retrieval down-weighted to 10% of the vector | 0.7244 | 0.8176 | 0.8477 | 0.7316 | 0.2627 | 0.4539 |

Contextual Retrieval is significantly worse on `paraphrase` - the slice it targets - at mean -0.0964, 95% CI [-0.1982, -0.0018]. See ARCHITECTURE.md for the mechanism; briefly, the section path is already indexed, so the generated sentence added vector mass without adding information.

Late interaction (ColBERT) was rejected the same way and is recorded in reports/colbert_eval.json and the component ablation.

## Net change, on the gold set these stages were measured against

| Metric | Start | Now | Change |
|---|---|---|---|
| nDCG@10 | 0.2760 | 0.7280 | +164% |
| recall@10 | 0.3967 | 0.8275 | +109% |
| recall@20 | 0.4900 | 0.8428 | +72% |
| MRR@10 | 0.2549 | 0.7302 | +186% |
| P@5 | 0.0960 | 0.2588 | +170% |

Questions returning zero relevant evidence: **26 -> 1** of 51.

## Final pipeline by question type

| Question kind | n | nDCG@10 | recall@20 |
|---|---|---|---|
| direct | 34 | 0.8205 | 0.9093 |
| multi_doc | 10 | 0.6116 | 0.8067 |
| paraphrase | 7 | 0.4455 | 0.5714 |

## Final pipeline by clinical topic

| Topic | n | nDCG@10 |
|---|---|---|
| anaemia | 2 | 1.0000 |
| mineral_bone | 1 | 1.0000 |
| risk | 5 | 0.9900 |
| blood_pressure | 4 | 0.8970 |
| treatment | 7 | 0.8151 |
| definition | 3 | 0.7445 |
| testing | 9 | 0.6817 |
| diet | 3 | 0.6583 |
| medication_safety | 7 | 0.6168 |
| referral | 3 | 0.5599 |
| staging | 5 | 0.5529 |
| monitoring | 2 | 0.3904 |

## Honesty notes

- The original pipeline reported hit@8 = 1.0 and MRR = 1.0. That evaluator counted a hit when any returned chunk carried a matching topic *label* - and the retriever boosted on those same labels, so it could not fail. The 0.276 baseline above is the same code measured against pinned gold chunks.
- nDCG is not comparable across the corpus rebuild: the gold set grew from 81 to 109 judgements as better chunks became available. recall@20 is normalised by gold count and is the fair cross-corpus number.
- The dev/test split exists because the original hand-tuned ranker scored 0.343 on dev and 0.134 on test - a 2.5x gap that was pure overfitting to eight smoke queries.
