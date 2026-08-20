# doc2query: a prediction, and the run that tested it

Written 2026-08-19, at **60% corpus coverage** (892 of 1479 chunks expanded, generation
interrupted by an API rate limit). Recorded now, ahead of the full-coverage measurement,
so the prediction below is on the record before the result that tests it.

## What was measured

Clinician-voice questions generated per chunk and indexed **separately** from the chunk
text, fused as one more signal (`w_doc2query`). Held-out chunk-first set, 46 questions,
split by phrasing style and by whether that question's gold chunk had been expanded yet.

| group | n | w=0.0 | w=0.4 | delta | 95% CI |
|---|---|---|---|---|---|
| clinician, gold expanded | 18 | 0.3265 | 0.5066 | **+0.1800** | **[+0.0667, +0.3084]** |
| clinician, not expanded | 8 | 0.5164 | 0.4539 | -0.0625 | [-0.2991, +0.1741] |
| echo, gold expanded | 12 | 0.8815 | 0.8580 | -0.0235 | [-0.1627, +0.0923] |
| echo, not expanded | 8 | 0.9278 | 0.8243 | -0.1035 | [-0.2253, -0.0066] |

The mechanism is visible per case rather than only in aggregate. `ho_29` asks "is a
flozin still worth starting?" - a word absent from both guidelines - and was a total
miss with the gold chunk outside the top 20. The generator wrote "Should I give a flozin
to someone with an eGFR of 30 and normal urine protein?" onto that chunk; gold now ranks
2nd, nDCG 0.000 -> 0.631. `ho_40` asks which tablets to hold during "a vomiting bug";
the generated "Which pills should patients pause when they catch a dehydrating bug?"
moves gold from rank 4 to rank 1.

## The prediction

Partial coverage is not a clean lower bound. An unexpanded chunk scores zero on this
signal, which weakens the feature; but an expanded *distractor* can outrank an
unexpanded *gold* chunk, which distorts it. The table above shows the harm falling
entirely in the two "not expanded" rows, and the only significant harm anywhere is
`echo, not expanded` at -0.1035.

So, at 100% coverage:

1. The `not expanded` groups cease to exist, and with them the -0.1035 harm.
2. The gain on clinician-phrased questions survives, because it does not depend on
   coverage of anything except the gold chunk itself.
3. The echo slice ends at or near parity - the -0.0235 on already-expanded echo
   questions is the honest estimate of this technique's cost, and its interval spans
   zero.

If instead the echo slice or the main 67-question gold set falls significantly at full
coverage, the technique is not free and the weight has to be traded off rather than
simply turned on. That is the outcome this prediction exists to be falsified by.

## Circularity: checked, clean

`evaluate_gold.py` excludes topic labels from scoring because the retriever boosts on
them, so scoring with them is circular. The corpus's template-built `hype_questions` are
keyed on exactly those labels, which is why they are not used here. The generated
expansions come from chunk *text*, so they should not reintroduce the problem - checked
rather than assumed, over the 2523 questions generated so far:

- identical to a template `hype_question`: **0**
- questions whose content words come entirely from the chunk's topic labels: **0**
- distinct questions: **2522 of 2523**, against the templates' 560 distinct of 3106

68% of the words in a generated question do not appear in its source chunk, which is the
term-injection effect the technique depends on and the reason it can reach vocabulary
the corpus does not contain.


## What is not yet known

- Whether the main gold set moves at all. It is question-first and its phrasing gap is
  small, so the expected effect there is near zero either way.
- The right `w_doc2query`. Only 0.4 has been tried, against 0.0.

## Outcome at 100% coverage (1475 of 1479 chunks, 4160 questions)

The prediction above was recorded before this was measured. All three parts held.

Held-out chunk-first set, 46 questions:

| w_doc2query | overall | clinician | echo | total misses |
|---|---|---|---|---|
| 0.0 | 0.6089 | 0.3849 | 0.9000 | 14 |
| 0.2 | 0.6548 | 0.4850 (+0.1001)* | 0.8755 (-0.0245) | 10 |
| 0.3 | 0.6791 | 0.5176 (+0.1327)* | 0.8889 (-0.0111) | 9 |
| **0.4** | **0.7014** | **0.5461 (+0.1612)\*** | **0.9033 (+0.0033)** | **8** |
| 0.5 | 0.6981 | 0.5594 (+0.1745)* | 0.8784 (-0.0216) | 8 |

\* 95% CI excludes zero. At w=0.4: clinician [+0.0747, +0.2581]; echo [-0.0803, +0.0730].

1. The harm predicted to be a coverage artefact disappeared. At 60% coverage the
   unexpanded echo group lost 0.1035 with an interval excluding zero; at full coverage
   the echo slice is +0.0033 with an interval spanning zero.
2. The clinician gain survived and grew, from +0.18 on a subgroup of 18 to +0.1612
   across all 26.
3. The echo slice ended at parity, as predicted.

Main gold set, 67 answerable question-first questions - the independent check that this
does not cost anything where the phrasing gap is small:

| w_doc2query | nDCG@10 | delta | 95% CI | recall@20 | total misses |
|---|---|---|---|---|---|
| 0.0 | 0.7125 | - | - | 0.8247 | 5 |
| 0.2 | 0.7236 | +0.0111 | [-0.0170, +0.0395] | 0.8433 | 4 |
| 0.3 | 0.7102 | -0.0023 | [-0.0380, +0.0311] | 0.8383 | 4 |
| 0.4 | 0.7088 | -0.0037 | [-0.0412, +0.0311] | 0.8275 | 3 |
| 0.5 | 0.6949 | -0.0176 | [-0.0561, +0.0193] | 0.8312 | 4 |

No weight moves it outside the noise floor, while total misses fall from 5 to 3 and
recall@20 rises. Shipped at **w_doc2query = 0.4**. End-to-end audit unchanged at 67/75
clean, no unsafe answers.

The phrasing gap - the thing this was built to close - goes from 0.515 to 0.357, and
unlike MedCPT it closes by lifting the weak side rather than lowering the strong one.

## Caveat on the weight

w was swept on the same 46 held-out questions the gain is reported from, so 0.4 is
chosen on the data it is scored on. The independent evidence that this is not
overfitting is the main gold set, which is flat across every weight from 0.2 to 0.5 -
the choice of weight is not what produces the result.

Coverage is now 1479 of 1479 chunks, 4172 questions. The four chunks left empty by the
trial run - including `Table 1| Criteria for chronic kidney disease`, gold for three
questions - were recovered by `--retry-empty`. On the complete index the main gold set
reads nDCG@10 0.7124, recall@10 0.8014, recall@20 0.8346, MRR@10 0.7121: nDCG is level
with the w=0 baseline to within 0.0001 and both recall figures are above it.
