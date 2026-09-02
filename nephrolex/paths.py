"""Every filesystem location the system uses, resolved once.

Before this module each core file computed the repository root itself as
`Path(__file__).resolve().parents[1]`. Eight modules carried that same expression,
which meant the constant was correct only as long as every one of them sat at exactly
that depth - so moving a module into a package directory would have silently
repointed it at `nephrolex/` instead of the repository, and the corpus, indexes and
gold set would have "disappeared" with no import error to explain why.

One definition, imported everywhere, makes directory depth stop being load-bearing.
"""

from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# The corpus lives beside the source in a checkout, but not in a container: there
# the code is baked into the image at /app and the operator's licensed guideline
# text is mounted separately, so that it can be backed up, replaced on a guideline
# revision, and kept out of the image. NEPHROLEX_DATA points at it; without the
# variable the layout is exactly as before.
DATA = Path(os.environ.get("NEPHROLEX_DATA") or (ROOT / "data"))
EVAL = ROOT / "eval"
REPORTS = ROOT / "reports"
SCRIPTS = ROOT / "scripts"

RAW_PDFS = DATA / "raw"
DOCLING = DATA / "docling"
CHUNKS = DATA / "chunks_v2"
INDEXES = DATA / "indexes"

CORPUS = CHUNKS / "retrieval_corpus.jsonl"
GOLD_SET = EVAL / "ckd_gold_eval.jsonl"
DENSE_INDEXES = INDEXES / "dense"
COLBERT_INDEX = INDEXES / "colbert"
