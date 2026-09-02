"""Assemble every measured evaluation run into one ablation table.

Reads reports/gold_eval_*.json and emits a markdown report. The point is that each
row is a run that actually happened and is reproducible from the recorded config,
not a number retyped from a slide.

Usage:
    python scripts/make_ablation_report.py
    python scripts/make_ablation_report.py --out reports/ABLATION.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

# Presentation order and human labels, oldest pipeline state first.
ORDER = [
    ("baseline_current_code", "Original: RRF + hand-written boost ladder"),
    ("normalized_lexical", "Normalised weighted fusion, boosts removed"),
    ("tuned_lexical", "+ index section paths & generated questions"),
    ("docling_lexical", "Corpus rebuilt from Docling (layout-aware)"),
    ("docling_v3", "+ query/expansion blending, length damping"),
    ("phaseA_final", "+ table context, mirror repair, range matching"),
    ("dense_bge_small", "+ dense: bge-small-en-v1.5 (33M)"),
    ("dense_bge_m3", "+ dense: bge-m3 (568M)"),
    ("rerank_bge_v2m3", "+ cross-encoder rerank: bge-reranker-v2-m3"),
    ("medembed_rerank", "swap dense -> MedEmbed-large (medical domain) + rerank"),
    ("medembed_rr30", "rerank depth 50 -> 30 (faster and better)"),
    ("filtered_corpus", "drop 84 front-matter / appendix chunks"),
    ("table_bbox_fix", "per-row table bboxes; drop abbreviation glossary"),
    ("tuned_weights", "fusion weights 0.30/0.20/0.30/0.20"),
    ("guidance_only", "restrict corpus to guidance sections"),
    ("figure_captions", "figure captions as retrievable chunks"),
    ("no_bm25", "drop BM25 from scoring (kept for candidate recall)"),
    ("round_weights", "weights 0.30/0.40/0.30, honest to the noise floor"),
]

# Techniques that were built, measured against the shipped configuration, and NOT
# kept. Reported separately from ORDER on purpose: these are not stages in the
# build-up, and folding them into that table would imply they were adopted. A
# well-known technique that did not work here is a result, not a gap - but only if
# the number is shown next to the one it lost to.
REJECTED = [
    ("contextual", "Contextual Retrieval, full generated context"),
    ("ctxdense", "Contextual Retrieval, generic words stripped"),
    ("blend90", "Contextual Retrieval down-weighted to 10% of the vector"),
]

METRICS = [
    ("ndcg_at_10", "nDCG@10"),
    ("recall_at_10", "recall@10"),
    ("recall_at_20", "recall@20"),
    ("mrr_at_10", "MRR@10"),
    ("p_at_5", "P@5"),
]


def load(tag: str) -> dict | None:
    path = REPORTS / f"gold_eval_{tag}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: float | None) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) else "-"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(REPORTS / "ABLATION.md"))
    args = parser.parse_args()

    runs = [(tag, label, load(tag)) for tag, label in ORDER]
    present = [(tag, label, report) for tag, label, report in runs if report]
    if not present:
        raise SystemExit("no gold_eval_*.json reports found")

    lines: list[str] = []
    lines.append("# Nephrolex retrieval ablation")
    lines.append("")
    # The stage table's own rows were measured against whatever gold set existed at the
    # time; quoting the oldest run's size as "the gold set" described a 50-question set
    # while the system was being evaluated on 67.
    current = load("current75") or present[-1][2]
    lines.append(
        f"Gold set today: {current['overall']['cases']} answerable questions with pinned "
        f"chunk-level relevance, plus scope and adversarial cases scored separately. "
        f"The historical rows below were measured against a {present[0][2]['overall']['cases']}-"
        "question set. `dev` is tuned on; `test` is held out."
    )
    lines.append("")

    lines.append(
        "**Historical, and not comparable to the current system.** Two things changed "
        "after these runs: scoring was corrected to treat KDIGO's two printings of a "
        "recommendation as one gold item, and the gold set grew from 51 to 67 answerable "
        "questions with harder paraphrase coverage. Neither can be applied backwards - "
        "these runs stored only their top 10, and their "
        "chunk IDs resolve against corpora since rebuilt (0% of the original baseline's IDs "
        "exist today), so a re-score would find no duplicates and report a meaningless "
        "\"no change\". For the shipped configuration under both metrics see the table below "
        "and the honesty notes in EVALUATION.md."
    )
    lines.append("")

    # ------------------------------------------------------------------ headline
    header = "| Pipeline stage | " + " | ".join(name for _, name in METRICS) + " | dev | test |"
    lines.append(header)
    lines.append("|" + "---|" * (len(METRICS) + 3))
    for _, label, report in present:
        overall = report["overall"]
        by_slice = report.get("by_slice", {})
        row = [label] + [fmt(overall.get(key)) for key, _ in METRICS]
        row.append(fmt(by_slice.get("dev", {}).get("ndcg_at_10")))
        row.append(fmt(by_slice.get("test", {}).get("ndcg_at_10")))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # ------------------------------------------------------- metric correction
    asbuilt, collapsed = load("current75_asbuilt"), load("current75")
    if asbuilt and collapsed:
        lines.append("## Shipped configuration, both metrics")
        lines.append("")
        lines.append("| Scoring | " + " | ".join(name for _, name in METRICS) + " |")
        lines.append("|" + "---|" * (len(METRICS) + 1))
        for label, report in (("as built", asbuilt), ("corrected", collapsed)):
            row = [label] + [fmt(report["overall"].get(key)) for key, _ in METRICS]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
        lines.append(
            "P@5 *falls* under the correction, because a duplicate in the top 5 no longer "
            "counts as a second relevant result. A correction that costs something on one "
            "metric is a correction rather than an adjustment."
        )
        lines.append("")

    # ---------------------------------------------------------------- rejected
    rejected = [(label, load(tag)) for tag, label in REJECTED]
    rejected = [(label, report) for label, report in rejected if report]
    if rejected:
        shipped = present[-1][2]
        lines.append("## Measured and rejected")
        lines.append("")
        lines.append(
            "Techniques tried against the shipped configuration and dropped. They are "
            "listed because a negative result with a mechanism is a stronger answer to "
            "\"why didn't you use X?\" than never having measured it. Every row here, "
            "baseline included, was measured on the 51-question gold set these experiments "
            "ran against - like for like. The current system scores differently on the "
            "larger set; see EVALUATION.md for that number."
        )
        lines.append("")
        lines.append("| Variant | " + " | ".join(name for _, name in METRICS) + " | paraphrase |")
        lines.append("|" + "---|" * (len(METRICS) + 2))
        row = ["**baseline: shipped config, same gold set as these rows**"] + [
            f"**{fmt(shipped['overall'].get(key))}**" for key, _ in METRICS
        ]
        row.append(f"**{fmt(shipped.get('by_kind', {}).get('paraphrase', {}).get('ndcg_at_10'))}**")
        lines.append("| " + " | ".join(row) + " |")
        for label, report in rejected:
            row = [label] + [fmt(report["overall"].get(key)) for key, _ in METRICS]
            row.append(fmt(report.get("by_kind", {}).get("paraphrase", {}).get("ndcg_at_10")))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
        lines.append(
            "Contextual Retrieval is significantly worse on `paraphrase` - the slice it "
            "targets - at mean -0.0964, 95% CI [-0.1982, -0.0018]. See ARCHITECTURE.md "
            "for the mechanism; briefly, the section path is already indexed, so the "
            "generated sentence added vector mass without adding information."
        )
        lines.append("")
        lines.append(
            "Late interaction (ColBERT) was rejected the same way and is recorded in "
            "reports/colbert_eval.json and the component ablation."
        )
        lines.append("")

    # -------------------------------------------------------------- improvement
    base, final = present[0][2]["overall"], present[-1][2]["overall"]
    lines.append("## Net change, on the gold set these stages were measured against")
    lines.append("")
    lines.append("| Metric | Start | Now | Change |")
    lines.append("|---|---|---|---|")
    for key, name in METRICS:
        before, after = base.get(key, 0), final.get(key, 0)
        pct = f"{(after - before) / before * 100:+.0f}%" if before else "-"
        lines.append(f"| {name} | {fmt(before)} | {fmt(after)} | {pct} |")
    lines.append("")
    lines.append(
        f"Questions returning zero relevant evidence: "
        f"**{base.get('total_miss')} -> {final.get('total_miss')}** of {final.get('cases')}."
    )
    lines.append("")

    # ------------------------------------------------------------- by question kind
    latest = present[-1][2]
    if latest.get("by_kind"):
        lines.append("## Final pipeline by question type")
        lines.append("")
        lines.append("| Question kind | n | nDCG@10 | recall@20 |")
        lines.append("|---|---|---|---|")
        for kind, stats in sorted(latest["by_kind"].items(), key=lambda kv: -kv[1]["ndcg_at_10"]):
            lines.append(
                f"| {kind} | {stats['cases']} | {fmt(stats['ndcg_at_10'])} | {fmt(stats['recall_at_20'])} |"
            )
        lines.append("")

    if latest.get("by_category"):
        lines.append("## Final pipeline by clinical topic")
        lines.append("")
        lines.append("| Topic | n | nDCG@10 |")
        lines.append("|---|---|---|")
        for name, stats in sorted(latest["by_category"].items(), key=lambda kv: -kv[1]["ndcg_at_10"]):
            lines.append(f"| {name} | {stats['cases']} | {fmt(stats['ndcg_at_10'])} |")
        lines.append("")

    lines.append("## Honesty notes")
    lines.append("")
    lines.append(
        "- The original pipeline reported hit@8 = 1.0 and MRR = 1.0. That evaluator "
        "counted a hit when any returned chunk carried a matching topic *label* - and the "
        "retriever boosted on those same labels, so it could not fail. The 0.276 baseline "
        "above is the same code measured against pinned gold chunks."
    )
    lines.append(
        "- nDCG is not comparable across the corpus rebuild: the gold set grew from 81 to "
        "109 judgements as better chunks became available. recall@20 is normalised by gold "
        "count and is the fair cross-corpus number."
    )
    lines.append(
        "- The dev/test split exists because the original hand-tuned ranker scored 0.343 on "
        "dev and 0.134 on test - a 2.5x gap that was pure overfitting to eight smoke queries."
    )

    out_path = Path(args.out)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
