"""Build the CKD gold retrieval evaluation set.

Gold relevance is pinned to real chunks in the retrieval corpus. Anchors describe
*what a chunk is*, not its ID, because chunk IDs encode page and type and so change
whenever the chunker changes - as happened when the corpus was rebuilt from Docling.
Supported anchor forms:

    "nice:1.5.5"                  NICE recommendation number
    "kdigo:Practice Point 5.1.1"  KDIGO recommendation / practice point label
    "text:markers of kidney"      distinctive substring, for unlabelled chunks
    "<legacy chunk id prefix>"    translated to a label automatically where possible

The resolver fails loudly if an anchor becomes ambiguous or stops matching, so the
gold set can never silently drift away from the corpus.

Grades follow standard graded-relevance conventions:
    2 = the chunk a clinician would cite to answer the question
    1 = useful supporting or partially-answering evidence
    0 = everything else (implicit)

Slices:
    dev  - tune on this
    test - held out; do not tune on this

Question kinds:
    direct      - phrased with guideline vocabulary
    paraphrase  - deliberate vocabulary mismatch with the source text
    multi_doc   - correct answer needs both KDIGO and NICE
    scope       - out of declared adult-CKD scope; system should flag/abstain
    unanswerable- not present in the corpus; system should abstain

Usage:
    python scripts/build_gold_eval.py
    python scripts/build_gold_eval.py --check   # validate only, write nothing
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


from newbieduo.paths import ROOT
CORPUS = ROOT / "data" / "chunks_v2" / "retrieval_corpus.jsonl"
MAX_MATCHES = 4
OUT = ROOT / "eval" / "ckd_gold_eval.jsonl"


# (id, slice, kind, category, query, [(chunk_id_prefix, grade), ...], note)
GOLD: list[tuple] = [
    # ---------------------------------------------------------------- definition
    (
        "def_nice_direct", "dev", "direct", "definition",
        "How does NICE define chronic kidney disease?",
        [("text:Abnormalities of kidney function or structure present for more than 3 months", 2)],
        "NICE definition lives in a narrative passage, not a numbered recommendation.",
    ),
    (
        "def_paraphrase", "test", "paraphrase", "definition",
        "What has to be true for someone to actually count as having long-term kidney "
        "disease rather than just one bad blood test?",
        [("text:Abnormalities of kidney function or structure present for more than 3 months", 2)],
        "No overlap with guideline vocabulary: no 'chronic', 'CKD', 'define', 'eGFR'.",
    ),
    (
        "kdigo_definition", "dev", "direct", "definition",
        "What is KDIGO's own formal definition of CKD?",
        [("text:CKD is defined as abnormalities of kidney structure", 2)],
        "Was unanswerable under the page-text corpus, where KDIGO's definition survived "
        "only inside a p193 reference-list chunk. The Docling rebuild recovers it.",
    ),

    # ------------------------------------------------------------------- staging
    (
        "gfr_g3b", "dev", "direct", "staging",
        "What eGFR range corresponds to GFR category G3b?",
        [("text:G3b; Moderately to severely decreased", 2),
         ("text:GFR category: G3b", 2),
         ("text:GFR category: G3a", 1),
         ("text:GFR category: G4", 1)],
        "",
    ),
    (
        "gfr_g4", "dev", "direct", "staging",
        "Which GFR category is described as severely decreased?",
        [("text:G4; Severely decreased", 2),
         ("text:GFR category: G4", 2)],
        "",
    ),
    (
        "gfr_g5", "test", "direct", "staging",
        "At what eGFR is a person classified as having kidney failure?",
        [("text:G5; Kidney failure", 2),
         ("text:GFR category: G5", 2)],
        "",
    ),
    (
        "staging_paraphrase", "test", "paraphrase", "staging",
        "My patient's filtration rate came back at 38. Which band does that put them in?",
        [("text:G3b; Moderately to severely decreased", 2),
         ("text:GFR category: G3b", 2)],
        "Requires mapping 38 into the 30-44 band without the query naming G3b or eGFR.",
    ),
    (
        "age_management", "dev", "direct", "staging",
        "Should CKD management be decided on the basis of age alone?",
        [("nice_ng203_ckd_p13_1_2_2", 2)],
        "",
    ),

    # ------------------------------------------------------------------- testing
    (
        "who_to_test", "dev", "direct", "testing",
        "Which adults should be offered testing for CKD?",
        [("nice_ng203_ckd_p11_1_1_21", 2)],
        "",
    ),
    (
        "acr_clinically_important", "dev", "direct", "testing",
        "What ACR level counts as clinically important proteinuria?",
        [("nice_ng203_ckd_p9_1_1_13", 2)],
        "",
    ),
    (
        "egfr_confirm", "dev", "direct", "testing",
        "Does a single low eGFR result need to be confirmed before diagnosing CKD?",
        [("nice_ng203_ckd_p8_1_1_8", 2)],
        "",
    ),
    (
        "egfr_caution", "test", "direct", "testing",
        "How should eGFR values of 60 ml/min/1.73 m2 or more be interpreted?",
        [("nice_ng203_ckd_p8_1_1_7", 2)],
        "",
    ),
    (
        "cystatin_when", "dev", "direct", "testing",
        "When should cystatin C-based eGFR be used instead of creatinine alone?",
        [("kdigo_2024_ckd_p36_recommendation_1_2_2_1", 2),
         ("kdigo_2024_ckd_p38_practice_point_1_2_2_6", 2)],
        "",
    ),
    (
        "gfr_initial_assessment", "dev", "direct", "testing",
        "What should be used for the initial assessment of GFR in adults?",
        [("kdigo_2024_ckd_p36_practice_point_1_2_2_1", 2),
         ("kdigo_2024_ckd_p34_recommendation_1_1_2_1", 1)],
        "",
    ),
    (
        "measured_gfr", "dev", "multi_doc", "testing",
        "When is a directly measured GFR needed rather than an estimate?",
        [("kdigo_2024_ckd_p36_practice_point_1_2_2_2", 2),
         ("nice_ng203_ckd_p8_1_1_9", 2),
         ("kdigo_2024_ckd_p38_practice_point_1_2_2_8", 1)],
        "",
    ),
    (
        "accurate_gfr_paraphrase", "test", "paraphrase", "testing",
        "We need a precise number before dosing chemotherapy. What does the guidance say?",
        [("nice_ng203_ckd_p8_1_1_9", 2)],
        "",
    ),
    (
        "ultrasound", "test", "direct", "testing",
        "Which people with CKD should be offered a renal ultrasound scan?",
        [("nice_ng203_ckd_p14_1_2_5", 2)],
        "",
    ),

    # ---------------------------------------------------------- risk / progression
    (
        "accel_progression", "dev", "direct", "risk",
        "How is accelerated progression of CKD defined in adults?",
        [("nice_ng203_ckd_p16_1_3_5", 2)],
        "",
    ),
    (
        "progression_rate", "dev", "direct", "risk",
        "How should the rate of progression of CKD be identified?",
        [("nice_ng203_ckd_p17_1_3_6", 2)],
        "",
    ),
    (
        "kfre_info", "dev", "direct", "risk",
        "Should adults with CKD be told their 5-year risk of needing renal replacement therapy?",
        [("nice_ng203_ckd_p21_1_5_1", 2),
         ("text:UK validation of the 4-variable", 1)],
        "",
    ),
    (
        "kfre_modality", "dev", "direct", "risk",
        "What kidney failure risk threshold is used to trigger modality education and planning?",
        [("kdigo_2024_ckd_p87_practice_point_2_2_3", 2),
         ("kdigo_2024_ckd_p41_practice_point_2_2_3", 2)],
        "Practice Point 2.2.3 is duplicated at p41 and p87; both are acceptable.",
    ),
    (
        "kfre_10pct", "test", "direct", "risk",
        "What 2-year kidney failure risk threshold is mentioned for multidisciplinary care?",
        [("kdigo_2024_ckd_p87_practice_point_2_2_2", 2)],
        "",
    ),

    # ------------------------------------------------------------------ referral
    (
        "referral_adults", "dev", "multi_doc", "referral",
        "When should an adult with CKD be referred for specialist assessment?",
        [("nice_ng203_ckd_p21_1_5_5", 2),
         ("kdigo_2024_ckd_p51_practice_point_5_1_1", 2)],
        "",
    ),
    (
        "referral_paraphrase", "test", "paraphrase", "referral",
        "At what point do I stop handling this in general practice and hand the patient over?",
        [("nice_ng203_ckd_p21_1_5_5", 2),
         ("kdigo_2024_ckd_p51_practice_point_5_1_1", 2)],
        "No 'refer', 'specialist', 'nephrology' or 'CKD' in the query.",
    ),
    (
        "referral_obstruction", "dev", "direct", "referral",
        "Who with CKD should be referred to urological services rather than nephrology?",
        [("nice_ng203_ckd_p22_1_5_8", 2)],
        "",
    ),

    # ---------------------------------------------------------------- monitoring
    (
        "monitoring_freq", "dev", "direct", "monitoring",
        "How often should eGFR be monitored in people with CKD?",
        [("nice_ng203_ckd_p15_1_3_4", 2),
         ("text:Minimum number of monitoring checks", 2),
         ("nice_ng203_ckd_p15_1_3_1", 1)],
        "",
    ),
    (
        "monitoring_paraphrase", "test", "paraphrase", "monitoring",
        "How many blood tests a year does someone with kidney disease actually need?",
        [("nice_ng203_ckd_p15_1_3_4", 2),
         ("text:Minimum number of monitoring checks", 2)],
        "",
    ),

    # ------------------------------------------------------------- blood pressure
    (
        "bp_target_low_acr", "dev", "direct", "blood_pressure",
        "What clinic blood pressure target applies to adults with CKD and an ACR under 70 mg/mmol?",
        [("nice_ng203_ckd_p24_1_6_1", 2)],
        "",
    ),
    (
        "bp_target_high_acr", "dev", "direct", "blood_pressure",
        "What blood pressure should be aimed for when ACR is 70 mg/mmol or more?",
        [("nice_ng203_ckd_p24_1_6_2", 2)],
        "",
    ),
    (
        "bp_kdigo", "dev", "direct", "blood_pressure",
        "What systolic blood pressure target does KDIGO suggest for adults with CKD?",
        [("kdigo_2024_ckd_p43_recommendation_3_4_1", 2),
         ("kdigo_2024_ckd_p97_recommendation_3_4_1", 1)],
        "",
    ),
    (
        "bp_conflict", "test", "multi_doc", "blood_pressure",
        "Do KDIGO and NICE agree on the systolic blood pressure target in CKD?",
        [("kdigo_2024_ckd_p43_recommendation_3_4_1", 2),
         ("nice_ng203_ckd_p24_1_6_1", 2),
         ("nice_ng203_ckd_p24_1_6_2", 2)],
        "Flagship conflict case: KDIGO <120 SBP vs NICE <140 / <130 by ACR. "
        "Retrieval must surface both guidelines or the answer cannot be correct.",
    ),

    # ----------------------------------------------------------------- treatment
    (
        "sglt2_kdigo", "dev", "multi_doc", "treatment",
        "Which patients with CKD should be started on an SGLT2 inhibitor?",
        [("kdigo_2024_ckd_p44_recommendation_3_7_1", 2),
         ("nice_ng203_ckd_p25_1_6_8", 1)],
        "",
    ),
    (
        "sglt2_continue", "dev", "direct", "treatment",
        "Should an SGLT2 inhibitor be stopped if the eGFR falls after starting it?",
        [("kdigo_2024_ckd_p44_practice_point_3_7_1", 2)],
        "",
    ),
    (
        "sglt2_withhold", "test", "direct", "treatment",
        "When should an SGLT2 inhibitor be temporarily withheld?",
        [("kdigo_2024_ckd_p44_practice_point_3_7_2", 2)],
        "",
    ),
    (
        "rasi_start", "dev", "multi_doc", "treatment",
        "When should RAS inhibitors be started in CKD with albuminuria?",
        [("kdigo_2024_ckd_p43_recommendation_3_6_1", 2),
         ("kdigo_2024_ckd_p43_recommendation_3_6_3", 2),
         ("nice_ng203_ckd_p24_1_6_5", 2),
         ("kdigo:Recommendation 3.6.2", 1)],
        "KDIGO Recommendation 3.6.2 is absent from Docling's parse; it is recovered by "
        "the column-aware PDF backfill in chunk_docling.py.",
    ),
    (
        "rasi_monitoring", "dev", "multi_doc", "medication_safety",
        "What should be checked after starting an ACE inhibitor or ARB in CKD?",
        [("kdigo_2024_ckd_p44_practice_point_3_6_2", 2),
         ("nice_ng203_ckd_p27_1_6_15", 2)],
        "",
    ),
    (
        "rasi_potassium_limit", "dev", "direct", "medication_safety",
        "Is there a potassium level above which a RAS antagonist should not be started?",
        [("nice_ng203_ckd_p27_1_6_16", 2)],
        "",
    ),
    (
        "hyperkalemia_manage", "test", "direct", "medication_safety",
        "How should hyperkalaemia associated with RAS inhibitor use be managed?",
        [("kdigo_2024_ckd_p44_practice_point_3_6_3", 2),
         ("kdigo_2024_ckd_p44_practice_point_3_6_5", 1)],
        "",
    ),
    (
        "creatinine_rise", "dev", "direct", "medication_safety",
        "What should be done if an adult's serum creatinine rises by 30% after starting an ARB?",
        [("nice_ng203_ckd_p28_1_6_23", 2)],
        "",
    ),
    (
        "acidosis", "dev", "multi_doc", "treatment",
        "When should oral sodium bicarbonate be considered in CKD?",
        [("nice_ng203_ckd_p51_1_12_8", 2),
         ("kdigo_2024_ckd_p45_practice_point_3_10_1", 2)],
        "",
    ),
    (
        "acidosis_ceiling", "test", "direct", "treatment",
        "Is there an upper limit to aim for when treating metabolic acidosis in CKD?",
        [("kdigo_2024_ckd_p45_practice_point_3_10_2", 2)],
        "",
    ),
    (
        "statin_cvd", "dev", "multi_doc", "treatment",
        "What lipid-lowering treatment applies to adults with CKD?",
        [("kdigo_2024_ckd_p47_recommendation_3_15_1_1", 2),
         ("nice_ng203_ckd_p28_1_6_24", 2),
         ("kdigo_2024_ckd_p47_recommendation_3_15_1_2", 1)],
        "",
    ),

    # ---------------------------------------------------------- medication safety
    (
        "nsaid", "dev", "multi_doc", "medication_safety",
        "Are NSAIDs safe to use in people with CKD?",
        [("nice_ng203_ckd_p18_1_3_10", 2),
         ("kdigo_2024_ckd_p49_practice_point_4_1_1", 2)],
        "",
    ),
    (
        "nephrotoxic_paraphrase", "test", "paraphrase", "medication_safety",
        "Patient wants to take ibuprofen for back pain - any concern given their kidneys?",
        [("nice_ng203_ckd_p18_1_3_10", 2),
         ("kdigo_2024_ckd_p49_practice_point_4_1_1", 2)],
        "'ibuprofen' does not appear in either guideline; requires NSAID generalisation.",
    ),
    (
        "otc_meds", "dev", "direct", "medication_safety",
        "What advice applies to over-the-counter medicines and herbal remedies in CKD?",
        [("kdigo_2024_ckd_p49_practice_point_4_1_3", 2)],
        "",
    ),

    # -------------------------------------------------------------------- diet
    (
        "protein_intake", "dev", "multi_doc", "diet",
        "What daily protein intake is suggested for adults with CKD?",
        [("kdigo_2024_ckd_p42_recommendation_3_3_1_1", 2),
         ("nice_ng203_ckd_p20_1_4_9", 2)],
        "",
    ),
    (
        "low_protein_paraphrase", "test", "paraphrase", "diet",
        "Is it a good idea to put someone with failing kidneys on a very low protein diet?",
        [("nice_ng203_ckd_p20_1_4_9", 2),
         ("kdigo_2024_ckd_p42_recommendation_3_3_1_1", 1)],
        "",
    ),
    (
        "dietary_advice", "dev", "direct", "diet",
        "What dietary advice should be given about potassium and phosphate in CKD?",
        [("nice_ng203_ckd_p20_1_4_7", 2),
         ("nice_ng203_ckd_p46_1_11_3", 1)],
        "",
    ),

    # ------------------------------------------------------------------- anaemia
    (
        "anaemia_esa_iron", "dev", "direct", "anaemia",
        "Can ESA therapy be started when absolute iron deficiency is present?",
        [("nice_ng203_ckd_p30_1_8_1", 2)],
        "",
    ),
    (
        "ferritin_limit", "test", "direct", "anaemia",
        "What serum ferritin ceiling applies when treating people with CKD with iron?",
        [("nice_ng203_ckd_p31_1_8_2", 2)],
        "",
    ),
    (
        "hyperparathyroidism", "dev", "direct", "mineral_bone",
        "How should clinically relevant secondary hyperparathyroidism be treated in CKD?",
        [("nice_ng203_ckd_p32_1_8_12", 2)],
        "",
    ),

    # ------------------------------------------------------- scope / abstention
    (
        "scope_pregnancy", "dev", "scope", "out_of_scope",
        "How should I manage CKD in a woman who is 20 weeks pregnant?",
        [],
        "Out of declared adult non-pregnant scope. Only 8 pregnancy-labelled chunks exist "
        "in the corpus. Correct behaviour: decline with a scope statement, not an answer.",
    ),
    (
        "scope_pregnancy_paraphrase", "test", "scope", "out_of_scope",
        "My patient is expecting and has stage 3 kidney disease - what do I do about her medication?",
        [],
        "Same scope test with no 'pregnant'/'pregnancy' token; a keyword scope gate fails here.",
    ),
    (
        "scope_children", "dev", "scope", "out_of_scope",
        "What ACR threshold triggers specialist referral for a 6-year-old?",
        [("nice_ng203_ckd_p22_1_5_6", 2)],
        "Paediatric evidence IS present, but is outside declared scope. Tests whether the "
        "scope gate fires independently of whether retrieval succeeds.",
    ),
    (
        "scope_dialysis_start", "test", "scope", "out_of_scope",
        "At what eGFR should I start my patient on haemodialysis?",
        [],
        "Dialysis initiation is not covered by NG203 or the KDIGO CKD guideline.",
    ),
    (
        "scope_emergency", "dev", "scope", "out_of_scope",
        "Potassium is 7.2 with ECG changes - what do I give right now?",
        [],
        "Acute emergency management. Must escalate, never answer from a chronic-care guideline.",
    ),

    # --------------------------------------------------------------- adversarial
    (
        "neg_unrelated", "dev", "unanswerable", "adversarial",
        "What is the recommended treatment for acute appendicitis?",
        [],
        "Nothing in the corpus is relevant. A non-empty confident answer is a failure.",
    ),
    (
        "neg_fabricated_threshold", "test", "unanswerable", "adversarial",
        "What is the ACR threshold of 500 mg/mmol used for in CKD staging?",
        [],
        "Premise is false - no such threshold exists. Tests whether the system corrects the "
        "premise or confabulates evidence for it.",
    ),

    # ================================================================ EXPANSION
    # Added after auditing why topic scores varied. Two features explained 47% of
    # per-query nDCG variance - gold chunks per query (r=-0.531) and query/gold
    # vocabulary overlap (r=+0.601) - and neither is a property of the topic. Topic
    # differences shrank 44% once both were controlled, so these cases are stratified
    # by *those* axes rather than by topic alone:
    #
    #   - every case pins 1-2 chunks, against 1-5 before, so annotation density stops
    #     confounding topic comparisons;
    #   - each weak topic gets a matched direct/paraphrase pair over the same evidence,
    #     which isolates vocabulary overlap from subject matter.
    #
    # Concentrated on the three topics whose residuals stayed negative after both
    # controls - monitoring (-0.236), referral (-0.121), staging (-0.117) - and on
    # paraphrase, which had only 7 cases and a 95% CI of +/-0.23.
    #
    # Every anchor below was chosen by reading the chunk text, never by running
    # retrieval and keeping what came back - that circularity is what made the
    # original evaluator unable to fail.

    # ---------------------------------------------------------------- monitoring
    (
        "mon_aki_followup", "dev", "direct", "monitoring",
        "How long should someone be monitored for CKD after an episode of acute kidney injury?",
        [("nice:1.1.25", 2)],
        "NICE gives an explicit duration; the paired paraphrase below uses none of its wording.",
    ),
    (
        "mon_aki_paraphrase", "test", "paraphrase", "monitoring",
        "Their kidneys took a hit in hospital last year and the bloods look fine now - "
        "do we still need to keep checking?",
        [("nice:1.1.25", 2)],
        "Matched pair with mon_aki_followup: same evidence, no shared vocabulary.",
    ),
    (
        "mon_acr_doubling", "dev", "direct", "monitoring",
        "What change in ACR on a repeat test exceeds laboratory variability?",
        [("kdigo:Practice Point 2.1.5", 2)],
        "Single-chunk answer, stated as a fold change rather than a threshold value.",
    ),
    (
        "mon_acr_paraphrase", "test", "paraphrase", "monitoring",
        "The urine protein result has gone up since last time - how much of a rise actually means something?",
        [("kdigo:Practice Point 2.1.5", 2)],
        "Matched pair with mon_acr_doubling.",
    ),
    (
        "mon_sglt2_frequency", "dev", "direct", "monitoring",
        "Does starting an SGLT2 inhibitor change how often CKD should be monitored?",
        [("kdigo:Practice Point 3.7.3", 2)],
        "Answer is negative - tests that the system reports what the guideline says "
        "rather than retrieving generic monitoring-frequency advice.",
    ),
    (
        "mon_haematuria_followup", "test", "direct", "monitoring",
        "How should persistent invisible haematuria without proteinuria be followed up?",
        [("nice:1.1.19", 2)],
        "Monitoring evidence that sits outside the eGFR/ACR vocabulary the topic usually uses.",
    ),

    # ------------------------------------------------------------------ referral
    (
        "ref_kfre_threshold", "dev", "direct", "referral",
        "What 5-year kidney failure risk supports referral to nephrology?",
        [("kdigo:Practice Point 2.2.1", 2)],
        "Risk-based referral criterion, distinct from the eGFR/ACR criteria already covered.",
    ),
    (
        "ref_kfre_paraphrase", "test", "paraphrase", "referral",
        "At what predicted chance of needing dialysis within five years should I get "
        "the kidney team involved?",
        [("kdigo:Practice Point 2.2.1", 2)],
        "Matched pair with ref_kfre_threshold.",
    ),
    (
        "ref_haematuria_cancer", "dev", "direct", "referral",
        "What should persistent invisible haematuria prompt investigation for?",
        [("nice:1.1.18", 2)],
        "Referral trigger that is not about kidney function at all.",
    ),
    (
        "ref_named_topic_paraphrase", "test", "paraphrase", "referral",
        "When does someone's kidney disease mean they need the specialist rather than us?",
        [("nice:1.5.5", 2)],
        "Deliberate contrast with referral_paraphrase, which scores 0.000 because it names "
        "no domain term at all ('stop handling this in general practice'). This one "
        "paraphrases the wording but keeps the topic, isolating vocabulary mismatch from "
        "topic underspecification.",
    ),

    # ------------------------------------------------------------------- staging
    (
        "stag_g1_direct", "dev", "direct", "staging",
        "What GFR range corresponds to category G1?",
        [("text:Row 1: GFR category: G1", 2)],
        "Single table row; G1/G2 were unpinned while G3b-G5 were already covered.",
    ),
    (
        "stag_g2_direct", "test", "direct", "staging",
        "Which GFR category covers 60 to 89 ml/min per 1.73 m2?",
        [("text:Row 2: GFR category: G2", 2)],
        "Reverse lookup: value to category, rather than category to value.",
    ),
    (
        "stag_g2_paraphrase", "test", "paraphrase", "staging",
        "Filtration came back at 72 with protein in the urine - where does that sit on the scale?",
        [("text:Row 2: GFR category: G2", 2)],
        "Matched pair with stag_g2_direct. The value is supplied by the question, which is "
        "the case the answer verifier had to be taught to allow.",
    ),
    (
        "stag_g1_paraphrase", "dev", "paraphrase", "staging",
        "Their filtration number is 95 - is that in the normal band?",
        [("text:Row 1: GFR category: G1", 2)],
        "Matched pair with stag_g1_direct.",
    ),

    # ------------------------------------------- paraphrase in well-served topics
    # The paraphrase slice was 7 cases with a 95% CI of [0.221, 0.678]. These add
    # coverage in topics that already score well, so the slice measures vocabulary
    # mismatch rather than the difficulty of whichever topics happened to be in it.
    (
        "bp_target_paraphrase", "test", "paraphrase", "blood_pressure",
        "How far down should I be pushing the top blood pressure number in someone "
        "with kidney disease?",
        [("kdigo:Recommendation 3.4.1", 2)],
        "blood_pressure scores 0.922 on direct phrasing; this tests the same evidence "
        "without the guideline's wording.",
    ),
    (
        "definition_paraphrase_2", "dev", "paraphrase", "definition",
        "How long do the kidney problems have to have been going on before it counts "
        "as the chronic kind?",
        [("text:CKD is defined as abnormalities of kidney structure", 2)],
        "Duration criterion only, phrased without 'chronic', 'CKD' or 'months'.",
    ),
    (
        "neg_wrong_guideline", "dev", "unanswerable", "adversarial",
        "What does NICE NG203 say about immunosuppression after kidney transplant?",
        [],
        "NG203 does not cover transplant immunosuppression.",
    ),
]


DOC_ALIASES = {"nice": "nice_ng203_ckd", "kdigo": "kdigo_2024_ckd"}

LEGACY_ID_RE = re.compile(r"^(kdigo_2024_ckd|nice_ng203_ckd)_p\d+_(.+)$")


def legacy_to_anchor(prefix: str) -> str | None:
    """Translate an old-style chunk-ID prefix into a label anchor.

    Old IDs looked like `nice_ng203_ckd_p21_1_5_5` or
    `kdigo_2024_ckd_p44_recommendation_3_7_1`. Table, parent and threshold chunks
    carry no label and cannot be translated - those are re-anchored by hand.
    """
    match = LEGACY_ID_RE.match(prefix)
    if not match:
        return None
    doc_id, rest = match.groups()
    if doc_id == "nice_ng203_ckd":
        number = re.match(r"^(\d+)_(\d+)_(\d+)(?:_|$)", rest)
        return f"nice:{'.'.join(number.groups())}" if number else None
    for token, title in (("recommendation_", "Recommendation"), ("practice_point_", "Practice Point")):
        if rest.startswith(token):
            number = re.match(r"^(\d+(?:_\d+)+)", rest[len(token):])
            if number:
                return f"kdigo:{title} {number.group(1).replace('_', '.')}"
    return None


def resolve_anchor(anchor: str, corpus: list[dict]) -> list[str]:
    """Return the chunk IDs an anchor matches."""
    if anchor.startswith("text:"):
        needle = anchor[5:].lower()
        return [c["id"] for c in corpus if needle in c["raw_text"].lower()]

    translated = anchor if ":" in anchor else legacy_to_anchor(anchor)
    if translated and ":" in translated:
        prefix, label = translated.split(":", 1)
        doc_id = DOC_ALIASES.get(prefix)
        if doc_id:
            return [
                c["id"]
                for c in corpus
                if c["metadata"].get("document_id") == doc_id
                and (c["metadata"].get("label") or "") == label
            ]

    return [c["id"] for c in corpus if c["id"].startswith(anchor)]


# Apparatus that must never be retrievable. Each of these reached the top of a real
# query at some point and was noticed only by looking at the screen - the gold set
# scored every one of them as neutral, because they were never gold *targets*. This
# turns "did a filter regress?" into a build-time failure instead of a demo surprise.
CORPUS_MUST_NOT_CONTAIN = [
    ("abbreviation glossary", r"angiotensin-converting enzyme inhibitor\(s\).{0,80}albumin-to-creatinine ratio"),
    ("table of contents", r"S\d{3}\s+[A-Z].{0,60}?S\d{3}\s+"),
    ("figure data grid", r"^Figure\s*\d+\|"),
    ("PICOS review scaffolding", r"PICOS|systematic review topics"),
    ("evidence-review bookkeeping", r"[Cc]itations screened"),
    ("document title page", r"NICE guideline Published:"),
    ("stub cross-reference", r"^Recommendation [\d.]+:?\s*to\s+[\d.]+\s*$"),
    ("work group / disclosure", r"WORK GROUP|EXECUTIVE COMMITTEE|Notice of rights"),
]


def check_corpus_hygiene(corpus: list[dict]) -> list[str]:
    """Assert the retrieval corpus carries no known apparatus."""
    problems = []
    for label, pattern in CORPUS_MUST_NOT_CONTAIN:
        rx = re.compile(pattern, re.M)
        hits = [c for c in corpus if rx.search(c["raw_text"])]
        if hits:
            problems.append(
                f"{label}: {len(hits)} chunk(s) still indexable, e.g. "
                + " ".join(hits[0]["raw_text"].split())[:70]
            )
    return problems


def load_corpus() -> list[dict]:
    with CORPUS.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Validate only; do not write.")
    args = parser.parse_args()

    corpus = load_corpus()
    ids = [record["id"] for record in corpus]

    hygiene = check_corpus_hygiene(corpus)

    errors: list[str] = []
    seen_case_ids: set[str] = set()
    rows: list[dict] = []

    for case_id, slice_name, kind, category, query, relevance, note in GOLD:
        if case_id in seen_case_ids:
            errors.append(f"{case_id}: duplicate case id")
        seen_case_ids.add(case_id)

        resolved: dict[str, int] = {}
        for anchor, grade in relevance:
            matches = resolve_anchor(anchor, corpus)
            if not matches:
                errors.append(f"{case_id}: anchor matched no chunk -> {anchor}")
            elif len(matches) > MAX_MATCHES:
                errors.append(
                    f"{case_id}: anchor is too broad ({len(matches)} matches) -> {anchor}\n"
                    + "".join(f"      {m}\n" for m in matches[:4])
                )
            else:
                # A guideline can legitimately restate a recommendation (KDIGO prints
                # its summary section and its chapter body); all copies are relevant.
                for chunk_id in matches:
                    resolved[chunk_id] = grade

        if kind not in {"scope", "unanswerable"} and not resolved:
            errors.append(f"{case_id}: answerable case has no gold chunks")

        rows.append(
            {
                "id": case_id,
                "slice": slice_name,
                "kind": kind,
                "category": category,
                "query": query,
                "relevance": resolved,
                "should_abstain": kind in {"scope", "unanswerable"},
                "note": note,
            }
        )

    if errors:
        print("GOLD SET VALIDATION FAILED\n")
        for error in errors:
            print("  " + error)
        raise SystemExit(1)

    dev = sum(1 for row in rows if row["slice"] == "dev")
    graded = sum(len(row["relevance"]) for row in rows)
    by_kind: dict[str, int] = {}
    for row in rows:
        by_kind[row["kind"]] = by_kind.get(row["kind"], 0) + 1

    print(f"cases            : {len(rows)}  (dev {dev} / test {len(rows) - dev})")
    print(f"graded judgements: {graded}")
    print(f"by kind          : {by_kind}")

    if args.check:
        print("\n--check: validated, nothing written.")
        return

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
