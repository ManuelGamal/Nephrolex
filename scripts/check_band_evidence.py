"""Does the row that answers a staging question actually reach the model?

The gold set already scored these cases, but nDCG@10 measures a ranked list of 20
while generation only ever sees the top SHOWN_EVIDENCE quotable chunks. A gold
chunk at rank 15 scores something and reaches nobody, so the aggregate metric was
satisfied by a pipeline that refused the question. This check closes that gap by
asserting against the evidence window generation actually reads.

Exit code 1 on any miss, so it can gate a change to the retrieval weights.
"""
import sys

from nephrolex.generation.answer import build_answer, SHOWN_EVIDENCE
from nephrolex import retrieval

# The models and depth scripts/demo_server.py serves with. Checking the
# library defaults instead would verify a path no user ever hits.
DENSE = "abhinand/MedEmbed-large-v0.1"
RERANK = "BAAI/bge-reranker-v2-m3"
TOP_K = 10

# query -> the chunk stating the band that answers it, from eval/ckd_gold_eval.jsonl
CASES = [
    ("My patient's filtration rate came back at 38. Which band does that put them in?",
     "kdigo_2024_ckd_p22_table_row_table_2_4", "38 -> G3b (30-44)"),
    ("My patient's eGFR came back at 38. Which band does that put them in?",
     "kdigo_2024_ckd_p22_table_row_table_2_4", "same question, eGFR wording"),
    ("Their filtration number is 95 - is that in the normal band?",
     "kdigo_2024_ckd_p22_table_row_table_2_1", "95 -> G1 (>=90)"),
    ("Which GFR category covers 60 to 89 ml/min per 1.73 m2?",
     "kdigo_2024_ckd_p22_table_row_table_2_2", "60-89 -> G2"),
]

failures = 0
print(f"evidence window generation reads: top {SHOWN_EVIDENCE} quotable chunks\n")
for query, required, note in CASES:
    result = build_answer(query, TOP_K, DENSE, RERANK, generate=False)
    # "quotes" is what generation reads. "evidence" is the panel shown beside the
    # answer, built from the ranked results; asserting on that measured the display
    # rather than the model's input, and reported a passing case as failing.
    shown = [
        str(e.get("chunk_id") or e.get("id") or "")
        for e in result.get("quotes", [])[:SHOWN_EVIDENCE]
    ]
    ok = required in shown
    failures += not ok
    print(f"[{'PASS' if ok else 'FAIL'}] {note}")
    print(f"        status={result['status']} conf={result.get('confidence')}")
    if not ok:
        print(f"        missing {required}")
        print(f"        shown:  {shown}")

print(f"\n{len(CASES) - failures}/{len(CASES)} staging questions reach the model with the band row.")
sys.exit(1 if failures else 0)
