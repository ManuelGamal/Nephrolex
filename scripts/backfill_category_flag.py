"""Write `defines_categories` into the retrieval corpus.

Metadata only. Chunk ids, order and text are asserted unchanged, so the dense vectors
(keyed by row position) and every lexical index stay valid and nothing needs re-embedding.

The rule itself lives in nephrolex/ingestion/chunk_docling.py, so a future rebuild from
the parsed documents reproduces the same flag rather than depending on this backfill.

    python scripts/backfill_category_flag.py --dry-run
    python scripts/backfill_category_flag.py
"""
import argparse
import json
import shutil
from datetime import datetime

from nephrolex.ingestion.chunk_docling import defines_categories
from nephrolex.paths import CORPUS, INDEXES

FIELD = "defines_categories"
RECORDS = INDEXES / "records.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Report, write nothing.")
    args = parser.parse_args()

    before = [json.loads(line) for line in CORPUS.open(encoding="utf-8")]
    flagged = [
        record["id"]
        for record in before
        if defines_categories(record["metadata"].get("section_path"))
    ]
    print(f"corpus       : {CORPUS}")
    print(f"chunks       : {len(before)}")
    print(f"would flag   : {len(flagged)} ({len(flagged) / len(before):.1%})")

    by_type: dict[str, int] = {}
    for record in before:
        if record["id"] in set(flagged):
            kind = record["metadata"].get("chunk_type", "?")
            by_type[kind] = by_type.get(kind, 0) + 1
    print(f"by chunk_type: {by_type}")

    if args.dry_run:
        print("\ndry run, nothing written")
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # records.json is what retrieval actually loads; the corpus is what a rebuild reads.
    # Both are patched, so the flag survives either path. Metadata only - tfidf.pkl and
    # the dense vectors are addressed by row position and are deliberately not rebuilt,
    # so the published retrieval numbers cannot move.
    for target in (CORPUS, RECORDS):
        backup = target.with_suffix(f"{target.suffix}.{stamp}.bak")
        shutil.copy2(target, backup)

        if target.suffix == ".jsonl":
            original = [json.loads(line) for line in target.open(encoding="utf-8")]
        else:
            original = json.loads(target.read_text(encoding="utf-8"))

        patched = [dict(record) for record in original]
        for record in patched:
            record["metadata"] = dict(record["metadata"])
            record["metadata"][FIELD] = defines_categories(
                record["metadata"].get("section_path")
            )

        assert len(patched) == len(original), "row count changed"
        for old, new in zip(original, patched):
            assert old["id"] == new["id"], f"id or order changed at {old['id']}"
            assert old["raw_text"] == new["raw_text"], f"raw_text changed at {old['id']}"
            assert old.get("text") == new.get("text"), f"text changed at {old['id']}"

        if target.suffix == ".jsonl":
            with target.open("w", encoding="utf-8", newline="\n") as handle:
                for record in patched:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written = [json.loads(line) for line in target.open(encoding="utf-8")]
        else:
            target.write_text(json.dumps(patched, ensure_ascii=False), encoding="utf-8")
            written = json.loads(target.read_text(encoding="utf-8"))

        hits = sum(1 for record in written if record["metadata"][FIELD])
        assert len(written) == len(original) and hits == len(flagged)
        print(f"  {target.name:<24} {len(written)} chunks, {hits} flagged "
              f"(backup {backup.name})")


if __name__ == "__main__":
    main()
