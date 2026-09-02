"""Build a held-out question set written chunk-first, then freeze it.

The existing gold set was written question-first: think of a clinical question, go find
the chunk that answers it. That direction cannot distinguish "the system failed" from
"the answer key was thin", which is how three correct answers came to score 0.000.

Here the chunk comes first. `scripts/sample_for_questions.py` draws a blind stratified
sample, the chunk text is read, and a question is written that the chunk answers. The
gold is therefore fixed before retrieval is ever run on the question, so the key cannot
be fitted to the output - not even accidentally, which is the failure mode that matters,
since nobody fits a key on purpose.

Rules followed while writing them, stated so they can be checked:

  - The question is answerable from the pinned chunk alone. No question needs a fact
    the chunk does not state.
  - The question does not reuse the chunk's distinctive wording where a clinician would
    not. "Which risk-factor domain covers drug-induced nephrotoxicity?" is fair;
    copying a whole recommendation back as the question is not a retrieval test.
  - A chunk that states no definite fact is dropped with a reason, not silently. Bare
    captions, pointers to other documents, health-economics prose, and NICE's own
    research questions carry nothing to ask about.
  - Where the corpus prints the same text twice, both printings are pinned, found by
    fingerprint rather than by hand.
  - Nothing was edited after seeing a retrieval result. This file is the record.

Usage:
    python scripts/build_heldout_set.py            # write eval/heldout_chunkfirst.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nephrolex.paths import CORPUS  # noqa: E402

OUT = ROOT / "eval" / "heldout_chunkfirst.jsonl"

# (question, chunk the answer was read from, the fact it states, topic)
QUESTIONS: list[tuple[str, str, str, str]] = [
    ("For which adults with type 2 diabetes does KDIGO suggest adding a nonsteroidal "
     "mineralocorticoid receptor antagonist?",
     "kdigo_2024_ckd_p44_recommendation_atom_recommendation_3_8_1_210",
     "eGFR >25, normal serum potassium, albuminuria >30 mg/g despite maximum tolerated RASi",
     "treatment"),
    ("What has to be taken into account when interpreting a serum creatinine level?",
     "kdigo_2024_ckd_p36_practice_point_practice_point_1_2_2_4_119",
     "dietary intake", "testing"),
    ("What should systems be in place to do for adults newly diagnosed with CKD?",
     "nice_ng203_ckd_p20_section_passage_self_management_ensure_that_systems_are_in_place_to_inform_67",
     "inform them of the diagnosis, enable shared decision making, support self-management",
     "self_management"),
    ("Why does KDIGO argue that systematic testing of people with risk factors is the only "
     "way to detect CKD early?",
     "kdigo_2024_ckd_p55_section_passage_1_1_1_detection_of_ckd_this_practice_point_promoting_ckd_de_307",
     "CKD progresses asymptomatically", "testing"),
    ("What is a prescribing cascade?",
     "kdigo_2024_ckd_p136_section_passage_in_the_context_of_good_drug_stewardship_healthcare_provider_827",
     "an adverse event misinterpreted as a new condition, treated with a further drug",
     "medication_safety"),
    ("Should someone with CKD and established ischaemic cardiovascular disease be on aspirin?",
     "kdigo_2024_ckd_p47_recommendation_atom_recommendation_3_15_2_1_246",
     "yes - oral low-dose aspirin for secondary prevention (1C)", "treatment"),
    ("When is etelcalcetide an option for secondary hyperparathyroidism in adults on "
     "haemodialysis?",
     "nice_ng203_ckd_p32_recommendation_atom_1_8_14_128",
     "when a calcimimetic is indicated but cinacalcet is not suitable", "mineral_bone"),
    ("Which GLP-1 receptor agonist should be chosen for someone with CKD?",
     "kdigo_2024_ckd_p106_practice_point_practice_point_3_9_1_649",
     "one with documented cardiovascular benefits", "treatment"),
    ("What does KDIGO recommend for adults with type 2 diabetes and CKD who have not reached "
     "their glycaemic target on metformin and an SGLT2 inhibitor?",
     "kdigo_2024_ckd_p45_recommendation_atom_recommendation_3_9_1_220",
     "a long-acting GLP-1 RA (1B)", "treatment"),
    ("Which medication should be considered for temporary discontinuation before elective "
     "surgery because of lactic acidosis if acute kidney injury occurs?",
     "kdigo_2024_ckd_p137_table_row_table_32_4",
     "metformin", "medication_safety"),
    ("What patient-related factors are associated with late referral for kidney replacement "
     "therapy planning?",
     "kdigo_2024_ckd_p142_table_row_table_35_1",
     "age, race, comorbid illness, aetiology, non-adherence, socioeconomic status",
     "referral"),
    ("What eGFR range defines GFR category G3a?",
     "kdigo_2024_ckd_p11_table_row_table_6",
     "45-59 ml/min per 1.73 m2, mildly to moderately decreased", "staging"),
    ("How much sodium a day does KDIGO suggest for people with CKD?",
     "kdigo_2024_ckd_p96_recommendation_atom_recommendation_3_3_2_1_573",
     "<2 g sodium (<90 mmol, <5 g sodium chloride) per day", "diet"),
    ("What level of protein intake should be avoided in adults with CKD at risk of progression?",
     "kdigo_2024_ckd_p42_practice_point_practice_point_3_3_1_1_177",
     ">1.3 g/kg body weight per day", "diet"),
    ("How should atrial fibrillation be looked for in people with CKD?",
     "kdigo_2024_ckd_p125_section_passage_identification_and_management_atrialfibrillation_can_be_asy_771",
     "opportunistic pulse-based screening, then a 12-lead ECG if the pulse is irregularly irregular",
     "cardiovascular"),
    ("At what 5-year risk of needing kidney replacement therapy should someone be referred to "
     "specialist kidney care?",
     "kdigo_2024_ckd_p140_figure_body_figure_48_20",
     ">3%-5% measured with a validated risk equation", "referral"),
    ("Which risk-factor domain for CKD covers drug-induced nephrotoxicity and radiation "
     "nephritis?",
     "kdigo_2024_ckd_p56_table_row_table_5_5", "iatrogenic", "risk"),
    ("At what serum potassium concentration should renin-angiotensin system antagonists be "
     "stopped?",
     "nice_ng203_ckd_p27_recommendation_atom_1_6_19_101",
     "6.0 mmol/litre or more, once other hyperkalaemia-promoting medicines are stopped",
     "medication_safety"),
    ("How should a positive reagent-strip albuminuria result be confirmed?",
     "kdigo_2024_ckd_p39_practice_point_practice_point_1_3_1_2_139",
     "quantitative laboratory measurement expressed as a ratio to urine creatinine",
     "testing"),
    ("When should an equation combining creatinine and cystatin C, or a measured GFR, be used "
     "for drug dosing?",
     "kdigo_2024_ckd_p134_practice_point_practice_point_4_2_3_814",
     "when more accuracy is needed - narrow therapeutic range, toxicity, or unreliable eGFRcr",
     "testing"),
]

# Batch two, seed 20260820. Same chunk-first construction, different phrasing rule.
#
# Batch one scored 0.90 partly because I wrote its questions echoing the chunk's own
# wording, so they carried the vocabulary of the passage they were pinned to. That is a
# real bias and it is the mirror image of the one in the question-first gold set. Chunk-
# first keying and clinician-voice phrasing are independent choices and batch one only
# made the first.
#
# These are phrased the way a clinician would actually ask, avoiding the chunk's
# distinctive terms wherever a real user would not reach for them - "a flozin", "sitting
# down all day", "a vomiting bug", "tablets" rather than "SGLT2i", "sedentary behavior",
# "acute dehydrating illness", "medical therapy". The key is still fixed before any
# retrieval runs, so the difference between the two batches isolates phrasing.
QUESTIONS_CLINICIAN: list[tuple[str, str, str, str]] = [
    ("I started an ACE inhibitor and the creatinine has climbed since. When is that a "
     "reason to stop it?",
     "kdigo_2024_ckd_p98_practice_point_practice_point_3_6_4_599",
     "stop only if SCr rises >30% within 4 weeks of starting or a dose increase", "treatment"),
    ("If someone is likely to deteriorate quickly, should I be testing them more often "
     "than usual?",
     "kdigo_2024_ckd_p81_practice_point_practice_point_2_1_2_473",
     "yes, when the measurement will change a therapeutic decision", "monitoring"),
    ("Is there anything different about looking after kidney patients in their early "
     "twenties?",
     "kdigo_2024_ckd_p52_practice_point_practice_point_5_3_1_2_1_289",
     "under-25s are a distinct high-risk group, partly from incomplete brain maturation",
     "referral"),
    ("The creatinine-based and cystatin-based estimates do not agree. Does the gap itself "
     "tell me anything?",
     "kdigo_2024_ckd_p68_practice_point_practice_point_1_2_2_7_385",
     "yes - both direction and magnitude of the difference can be informative", "testing"),
    ("How should I work out someone's chance of ending up needing dialysis?",
     "kdigo_2024_ckd_p84_recommendation_atom_recommendation_2_2_1_494",
     "an externally validated risk equation, in CKD G3-G5 (1A)", "risk"),
    ("Is there a nutrition screening tool that also takes in mobility and mental state?",
     "kdigo_2024_ckd_p146_table_row_table_39_3",
     "Mini Nutrition Assessment; 12-14 points indicates normal nutrition", "diet"),
    ("Which way of estimating kidney function is most accurate in someone with severe "
     "obesity?",
     "kdigo_2024_ckd_p64_table_row_table_8_5", "eGFRcr-cys", "testing"),
    ("What should I be telling a kidney patient about sitting down all day?",
     "kdigo_2024_ckd_p92_practice_point_practice_point_3_2_2_2_538",
     "advise them to avoid sedentary behaviour", "diet"),
    ("Filtration is 35 and there is barely any protein in the urine. Is a flozin still "
     "worth starting?",
     "kdigo_2024_ckd_p44_recommendation_atom_recommendation_3_7_3_208",
     "yes - suggested for eGFR 20-45 with ACR <200 mg/g (2B)", "treatment"),
    ("A 40-year-old with kidney disease and diabetes - do they need a statin?",
     "kdigo_2024_ckd_p119_recommendation_atom_recommendation_3_15_1_3_734",
     "yes, 18-49 with diabetes is one of the listed indications (2A)", "treatment"),
    ("Can I just work from the creatinine result itself instead of converting it?",
     "kdigo_2024_ckd_p71_recommendation_atom_recommendation_1_2_4_1_409",
     "no - use a validated GFR estimating equation rather than the marker alone (1D)",
     "testing"),
    ("Who with kidney disease should be started on an SGLT2 inhibitor?",
     "kdigo_2024_ckd_p99_recommendation_atom_recommendation_3_7_2_610",
     "eGFR >=20 with ACR >=200 mg/g, or heart failure at any albuminuria (1A)", "treatment"),
    ("Potassium has come back at 5.8 in someone on finerenone. What now?",
     "kdigo_2024_ckd_p105_figure_body_figure_26_16",
     "K+ >5.5: hold finerenone, adjust diet or concomitant drugs, recheck", "medication_safety"),
    ("Why will the guideline not back urate-lowering treatment in someone with a high "
     "urate but no symptoms?",
     "kdigo_2024_ckd_p118_section_passage_rationale_there_is_insufficient_evidence_to_recommend_the_u_726",
     "insufficient evidence; benefits for slowing progression are unclear", "treatment"),
    ("A kidney patient keeps getting gout attacks. Should the urate be treated?",
     "kdigo_2024_ckd_p115_recommendation_atom_recommendation_3_14_1_704",
     "yes - offer urate-lowering intervention for symptomatic hyperuricaemia (1C)", "treatment"),
    ("Do I need to chase down why the kidneys are failing, or is managing it enough?",
     "nice_ng203_ckd_p14_recommendation_atom_1_2_3_43",
     "agree a plan to establish the cause, especially if it may be treatable", "testing"),
    ("How much salt a day is right for a child of eleven?",
     "kdigo_2024_ckd_p97_table_row_table_22_5", "1.2 g/day for ages 9-13", "diet"),
    ("What is the high-dose intravenous iron regimen for the first month on dialysis?",
     "nice_ng203_ckd_p39_table_row_table_3_1",
     "600 mg iron sucrose divided equally over 3 haemodialysis sessions", "anaemia"),
    ("Which form of vitamin D should I give a kidney patient who is deficient?",
     "nice_ng203_ckd_p50_recommendation_atom_1_12_5_211",
     "colecalciferol or ergocalciferol", "mineral_bone"),
    ("Which tablets should someone with kidney disease hold when they have a vomiting bug?",
     "kdigo_2024_ckd_p136_section_passage_sick_day_rules_have_been_endorsed_as_useful_guidance_to_peop_829",
     "SADMANS - sulfonylureas, ACEi, diuretics/DRI, metformin, ARBs, NSAIDs, SGLT2i",
     "medication_safety"),
    ("Is an estimate ever not good enough, so that a proper measurement is needed?",
     "kdigo_2024_ckd_p65_table_row_table_9_3",
     "eGFR is not sufficiently accurate for all clinical situations; measured GFR is",
     "testing"),
    ("How do I dose drugs in someone whose kidney function is swinging about?",
     "kdigo_2024_ckd_p134_practice_point_practice_point_4_2_5_819",
     "adapt dosing when GFR, non-GFR determinants or volume of distribution are unsteady",
     "medication_safety"),
    ("For stable angina in someone with kidney disease, is stenting better than tablets?",
     "kdigo_2024_ckd_p125_section_passage_rationale_the_totality_of_the_evidence_from_the_ckd_specifi_769",
     "no net difference between conservative medical therapy and an invasive strategy",
     "cardiovascular"),
    ("Is it safe to combine an ACE inhibitor with an ARB in kidney disease?",
     "kdigo_2024_ckd_p98_recommendation_atom_recommendation_3_6_4_595",
     "no - avoid any combination of ACEi, ARB and direct renin inhibitor (1B)",
     "medication_safety"),
    ("What can throw off a urine albumin result?",
     "kdigo_2024_ckd_p39_practice_point_practice_point_1_3_1_3_140",
     "several factors; order confirmatory tests as indicated (Table 16)", "testing"),
    ("What should be done when potassium goes above 5.5?",
     "kdigo_2024_ckd_p113_figure_caption_figure_32_687",
     "Figure 32 sets out the actions to manage hyperkalaemia in CKD", "medication_safety"),
]

# A chunk other than the sampled one can state the same fact. Each pattern below is
# written from that question's `fact` string - recorded before any retrieval ran - and
# matches only a chunk that states the fact outright. Questions absent from this table
# have a fact stated in exactly one place, checked by sweeping for its key term and
# finding nothing else.
ANSWER_PATTERNS: dict[str, str] = {
    # "45-59 ml/min per 1.73 m2, mildly to moderately decreased"
    "ho_12": r"Row \d+:\s*(GFR category:\s*)?G3a\b",
    # "opportunistic pulse-based screening, then a 12-lead ECG"
    "ho_15": r"opportunistic pulse-based screening",
    # ">3%-5% measured with a validated risk equation"
    "ho_16": r">\s*3%\s*-\s*5%\s*5-year risk|3%\s*-\s*5% 5-year risk of requiring KRT",
    # "one with documented cardiovascular benefits"
    "ho_08": r"choice of GLP-1 RA should prioritize",
    # "dietary intake" affecting interpretation of serum creatinine
    "ho_02": r"Interpretation of SCr level requires consideration of dietary intake",
}

# Sampled but not turned into a question, with the reason. Recorded so the drop rate and
# its causes are visible rather than inferred from a missing count.
DROPPED: list[tuple[str, str]] = [
    # --- batch two, seed 20260820
    ("kdigo_2024_ckd_p128_section_passage_key_information_values_and_preferences_high_value_on_the_u_785",
     "values-and-preferences discussion of NOACs, no definite fact to ask for"),
    ("kdigo_2024_ckd_p85_section_passage_key_information_values_and_preferences_the_work_group_judg_502",
     "values-and-preferences discussion of risk prediction, deliberately non-prescriptive"),
    ("nice_ng203_ckd_p54_section_passage_as_part_of_the_2021_update_the_guideline_committee_made_18_226",
     "meta - counts the committee's research recommendations, states no clinical fact"),
    ("kdigo_2024_ckd_p58_section_passage_1_1_4_evaluation_of_cause_education_on_the_value_of_establi_334",
     "implementation prose about education and resourcing"),
    ("nice_ng203_ckd_p63_section_passage_why_the_committee_made_the_recommendations_the_committee_ma_258",
     "records that a research recommendation was made; the answer does not exist yet"),
    ("kdigo_2024_ckd_p117_section_passage_key_information_the_ertreview_identified_25_studies_26_pub_722",
     "evidence-review bookkeeping - counts of studies and publications"),
    # --- batch one, seed 20260819

    ("kdigo_2024_ckd_p22_figure_caption_table_3_60",
     "bare caption - 'Table 3| Albuminuria categories in chronic kidney disease' states no fact"),
    ("nice_ng203_ckd_p56_section_passage_what_is_the_most_clinical_and_cost_effective_frequency_of_re_236",
     "is itself a NICE research question, not an answer to one"),
    ("kdigo_2024_ckd_p103_section_passage_key_information_resource_use_and_costs_health_economic_ana_629",
     "health-economics prose with no definite clinical fact to ask for"),
    ("kdigo_2024_ckd_p43_section_passage_special_considerations_pediatric_considerations_the_work_g_189",
     "a pointer - 'we highlight the following guidance' - the guidance itself is elsewhere"),
    ("kdigo_2024_ckd_p72_section_passage_key_information_materials_cannot_be_applied_to_settings_wit_415",
     "values-and-preferences discussion, deliberately non-prescriptive"),
    ("kdigo_2024_ckd_p52_practice_point_practice_point_5_2_3_1_279",
     "'use evidence-informed management strategies' - too general to have a definite answer"),
    ("nice_ng203_ckd_p16_table_row_table_3",
     "CORPUS DEFECT, not a question-writing problem: verified against the source PDF, this is "
     "NICE's monitoring-frequency table and the numbers are correct (G3a row = 1, 1, 2), but "
     "all six rows carry only the ACR column headers - the GFR row label is lost, so no row "
     "can be attributed to a category. The chunk cannot answer a question about it."),
]


def fingerprint(text: str) -> str:
    """Identity of a chunk's text. Full text, deliberately.

    Truncating to a prefix - as `evaluation/evaluate_gold.py::_fingerprints` does at 160
    characters - makes every row of a wide table share one identity, because a table_row
    chunk repeats the table's header before its own payload. It pinned all six rows of
    Table 32 as equally correct answers to a question about metformin.
    """
    return re.sub(r"[^a-z0-9]", "", text.lower())


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()

    with CORPUS.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    by_id = {r["id"]: r for r in records}

    prints: dict[str, list[str]] = {}
    for record in records:
        prints.setdefault(fingerprint(record["raw_text"]), []).append(record["id"])

    tagged = ([(q, "echo") for q in QUESTIONS]
              + [(q, "clinician") for q in QUESTIONS_CLINICIAN])
    cases = []
    for i, ((query, chunk_id, fact, topic), style) in enumerate(tagged, 1):
        record = by_id.get(chunk_id)
        if record is None:
            raise SystemExit(f"question {i} pins a chunk not in the corpus: {chunk_id}")
        # Both printings of the same text are equally correct answers.
        relevance = {cid: 2 for cid in prints[fingerprint(record["raw_text"])]}

        # And so is any *other* chunk stating the same fact. The first version of this
        # file pinned only the sampled chunk, which reproduced the exact defect the
        # held-out set was built to expose: asked what range defines G3a, the system
        # returned KDIGO Table 2's G3a row at rank 1 - correct - and scored 0.356,
        # because the sampler had excluded that row for being pinned elsewhere and left
        # only the p11 reprint in the key.
        #
        # The sweep pattern is derived from `fact`, which was written from the chunk
        # before any retrieval ran, and it is applied to all 20 questions uniformly.
        pattern = ANSWER_PATTERNS.get(f"ho_{i:02d}")
        if pattern:
            for other in records:
                if other["id"] in relevance:
                    continue
                if re.search(pattern, " ".join(other["raw_text"].split()), re.I):
                    relevance[other["id"]] = 2

        cases.append({
            "id": f"ho_{i:02d}_{topic}",
            "query": query,
            "kind": f"heldout_{style}",
            "category": topic,
            "slice": "heldout",
            "phrasing": style,
            "should_abstain": False,
            "relevance": relevance,
            "note": f"written from {chunk_id}; states: {fact}",
        })

    with Path(args.out).open("w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    multi = sum(1 for c in cases if len(c["relevance"]) > 1)
    print(f"{len(cases)} held-out questions -> {args.out}")
    print(f"  {multi} pin more than one chunk (the corpus prints the text twice)")
    print(f"  {len(DROPPED)} sampled chunks dropped:")
    for chunk_id, reason in DROPPED:
        print(f"    {chunk_id[:56]}\n      {reason}")


if __name__ == "__main__":
    main()
