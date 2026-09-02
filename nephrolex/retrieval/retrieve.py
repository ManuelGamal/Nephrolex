"""Hybrid CKD guideline retrieval.

Scoring model
-------------
Every retriever (TF-IDF, dense) produces raw scores on its own arbitrary scale.
Those are min-max normalised into [0, 1] over the shared candidate pool and then
combined with explicit, inspectable weights:

    relevance = w_tfidf*tfidf + w_dense*dense + w_meta*metadata
    score     = relevance * chunk_type_prior * quality_penalty

BM25 was removed entirely
-------------------------
It first lost its scoring weight, because adding it back at w=0.30 costs 0.032
nDCG@10 with an interval excluding zero. It was kept in the candidate pool on the
assumption that it still earned its place by recall, and this file said so.

That assumption was then measured and was false. Removing BM25 from the pool as
well is a small significant *improvement* - +0.0010 nDCG@10, 95% CI [+0.0000,
+0.0031] - and recall@20 rises from 0.8371 to 0.8445. It contributed nothing to
ranking and nothing to recall, so it is gone rather than disabled: one fewer index
to build and one fewer thing to explain.

The likely mechanism is interaction with query expansion: the expansion adds up to
a dozen synonyms and BM25's IDF weighting rewards documents matching many rare
ones, which is how a gadolinium practice point outranked the GFR staging table for
"filtration rate 38, which band?".

This is not dense-only: TF-IDF remains as the lexical component, so the pipeline is
still hybrid.

A two-guideline coverage pass used to live here, inserting evidence from whichever
guideline was missing from the results. It is gone. Enabled against the shipped
configuration it changed no ranking on any of the 51 gold questions - including the
10 that require both KDIGO and NICE, the case it existed for. Diversification and the
chunk-type priors already surface both documents, so it was complexity with a
measured effect of exactly zero.

Weights
-------
0.21 lexical / 0.29 dense / 0.21 metadata / 0.29 doc2query, summing to one.

Declared as shares and asserted to sum to 1 at import. They were 0.30/0.40/0.30/0.40 -
a raw total of 1.40 that the fold divided away - which computed the same ranking and
invited the same question every time anyone read it.

The fold still divides by the sum of the *active* weights, and that is load-bearing
rather than redundant: a signal whose index is missing contributes nothing and is also
excluded from the denominator, so an absent doc2query index leaves the other three at
exactly the ratios they had before it existed, instead of silently rescaling every
score.

The round numbers are deliberate. Two sweeps - ten fusion configurations, then five
weight splits - found every setting inside the bootstrap noise floor at this sample
size; no interval excluded zero. The previous values normalised to 0.29/0.43/0.29,
which implied a precision the measurement does not support. These were best on both
the full set (0.7280) and the held-out slice (0.6751), and they say the only thing
the data actually supports: dense carries somewhat more weight than either lexical
or metadata, and those two are comparable.

What the sweeps *did* establish, outside the noise:

  - Starving dense retrieval costs 0.046 nDCG@10, significant. It is the component
    that cannot be cheaply reduced.
  - Reciprocal Rank Fusion instead of this convex combination costs 0.029, with an
    interval spanning zero - so the choice is defensible either way, and is measured
    rather than inherited.

This replaces an earlier design that fused with RRF (max contribution ~0.016 per
retriever) and then *added* hand-written boosts of up to +3.3. In that design the
retrievers determined roughly 2% of the final ranking and a rule ladder
determined the rest, so measuring an embedding model or a reranker was
impossible - their contribution was numerically drowned. Everything below is
query-independent: there are no rules keyed to specific evaluation questions.

Usage:
    python scripts/retrieve.py "When should an adult with CKD be referred?" --top-k 8
    python scripts/retrieve.py "..." --dense-model BAAI/bge-m3 --reranker-model BAAI/bge-reranker-v2-m3
    python scripts/retrieve.py "..." --w-dense 0.5 --w-tfidf 0.2 --explain
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


from nephrolex.paths import ROOT
INDEX_DIR = ROOT / "data" / "indexes"

# Fusion weights, declared as shares of the score and required to sum to 1.
#
# They were relative before - 0.30/0.40/0.30/0.40, a raw total of 1.40 that the fold
# divided away. That was arithmetically fine and rhetorically bad: the first question
# anyone asks of four numbers printed next to each other is whether they sum to one, and
# "they are relative, we normalise later" is a worse answer than not needing one. These
# are the same model expressed as the shares it already computed.
#
# The assertion is the point. A future signal added without rebalancing would silently
# rescale every other weight; this makes that a startup failure instead.
W_TFIDF = 0.21
W_DENSE = 0.29
W_META = 0.21
W_DOC2QUERY = 0.29

# Additive weight on the numeric band signal.
#
# NUMERIC_BAND_PRIOR below applies the same signal multiplicatively, which scales
# whatever relevance the lexical and dense signals already assigned. That is exactly
# backwards for the case the signal exists to serve: "eGFR 38, which category?" shares
# no vocabulary with the row that answers it, because the token 38 never appears in
# "30 - 44". Low relevance times a bounded multiplier stays low, so the answer row sat
# at rank 15 while the top 8 went to the model. An additive term pays the chunk in
# absolute score instead, which is what a quantity-anchored band containment deserves.
#
# Measured and rejected. Swept on the 67-case gold set against a w_band=0 control that
# reproduced 0.7250 exactly, dense and reranker both confirmed loaded:
#
#     w_band   nDCG@10   staging   total_miss   paired bootstrap vs 0.00
#     0.00     0.7250    0.5722    4            control
#     0.10     0.7119    0.5670    4            -0.0131  [-0.0306, +0.0010]  noise
#     0.20     0.6875    0.5559    6            -0.0375  [-0.0708, -0.0074]  worse
#     0.30     0.6611    0.5312    8            -0.0638  [-0.1073, -0.0254]  worse
#
# It fails on the slice it was written for: staging falls too. The multiplicative prior
# is a conjunction - relevant AND band-matching - while an additive term is a
# disjunction, so every lab-variation table stating GFR bands (Tables 23, 24, 29) gets
# promoted alongside the one row that answers the question. For a signal this
# unselective, conjunction is the correct form.
#
# Kept at 0.0, which is bit-identical to the multiplicative-only path, so the knob
# remains available for a future sweep without changing today's behaviour.
W_BAND = 0.0

# Which query text the doc2query index is scored against.
#   "raw"      - the shipped behaviour; loses the register gap it exists to close
#   "expanded" - as tfidf and metadata already do
#   "max"      - per-chunk max, so expansion can only add signal where raw found none
# Resolved at call time so a sweep can flip it without rebinding the default.
#
# Measured on the 67-case gold set, control reproducing 0.7250 exactly:
#
#     mode       nDCG@10   staging   paired bootstrap vs raw
#     raw        0.7250    0.5722    control
#     expanded   0.7215    0.5786    -0.0035  [-0.0224, +0.0115]  inside noise
#
# Inside the noise floor overall, but it moved 15 of 67 cases and took
# gfr_initial_assessment from 0.9502 to 0.4841 - the added "category
# classification range" terms promote table rows over the practice points that
# answer a definitional question. "max" recovered only part of that (0.6697) and
# fixed no additional case. An aggregate inside the noise floor is not a licence
# to ship a half-point regression, so this stays on "raw"; the staging failure it
# was chasing is fixed in the evidence window instead, where it belongs.
DOC2QUERY_QUERY_MODE = "raw"

assert abs(W_TFIDF + W_DENSE + W_META + W_DOC2QUERY - 1.0) < 1e-9, (
    "fusion weights must sum to 1"
)

CANDIDATE_POOL = 120


# Domain synonym expansion. This is general CKD vocabulary knowledge applied to
# any query that mentions the trigger term - not per-question tuning.
QUERY_EXPANSIONS = {
    "definition": ["defined as", "markers of kidney damage", "abnormalities of kidney structure or function", "present for at least 3 months"],
    "defined": ["definition", "markers of kidney damage", "present for at least 3 months"],
    "diagnosis": ["definition", "markers of kidney damage", "albuminuria"],
    "albuminuria": ["ACR", "UACR", "urine albumin-creatinine ratio", "proteinuria"],
    "acr": ["albuminuria", "UACR", "urine albumin-creatinine ratio", "proteinuria"],
    "proteinuria": ["albuminuria", "ACR", "UACR"],
    "egfr": ["GFR", "glomerular filtration rate", "GFR category"],
    "gfr": ["eGFR", "estimated glomerular filtration rate"],
    "kidney failure": ["G5", "kidney failure risk", "end-stage"],
    "kidney function": ["eGFR", "GFR", "creatinine", "cystatin C"],
    "renal function": ["eGFR", "GFR", "creatinine", "cystatin C"],
    "filtration rate": ["eGFR", "GFR", "glomerular filtration rate"],
    "staging": ["G1", "G2", "G3a", "G3b", "G4", "G5", "A1", "A2", "A3", "CGA", "GFR category"],
    "stage": ["G1", "G2", "G3a", "G3b", "G4", "G5", "A1", "A2", "A3", "CGA", "GFR category"],
    "category": ["GFR category", "albuminuria category", "CGA", "classification"],
    "band": ["GFR category", "classification", "range"],
    "refer": ["referral criteria", "specialist kidney care", "nephrology", "specialist assessment"],
    "referral": ["specialist kidney care", "nephrology", "specialist assessment"],
    "nephrology": ["specialist kidney care", "referral criteria"],
    "monitor": ["monitoring frequency", "follow-up", "repeat testing"],
    "monitoring": ["frequency", "follow-up", "repeat testing"],
    "blood test": ["monitoring", "eGFR", "frequency of monitoring"],
    "blood pressure": ["hypertension", "systolic", "ACE inhibitor", "ARB", "RAS inhibitor"],
    "hypertension": ["blood pressure", "systolic", "ACE inhibitor", "ARB"],
    "sglt2": ["SGLT2 inhibitor", "sodium-glucose cotransporter-2 inhibitor"],
    "kidney failure risk": ["KFRE", "Kidney Failure Risk Equation", "5-year risk"],
    "ibuprofen": ["NSAID", "non-steroidal anti-inflammatory", "nephrotoxic"],
    "painkiller": ["NSAID", "analgesic", "nephrotoxic"],
    "dialysis": ["kidney replacement therapy", "renal replacement therapy"],
    "pregnant": ["pregnancy"],
    "expecting": ["pregnancy", "pregnant"],
    "child": ["children", "young people", "paediatric"],
    "diet": ["dietary", "protein intake", "nutrition"],
}


# Prior on chunk types: a numbered recommendation is a better citation target than
# a broad section blob. Applied multiplicatively, and deliberately mild - a prior
# should nudge ties, not override relevance.
CHUNK_TYPE_PRIOR = {
    "recommendation_atom": 1.15,
    "practice_point": 1.13,
    "table_row": 1.10,
    "figure_caption": 1.05,
    # A rebuilt figure body carries the criteria the caption only names - Figure 48
    # is the whole referral criteria set - so it ranks with table rows, not prose.
    "figure_body": 1.10,
    "threshold_fact": 1.00,
    "section_passage": 0.88,
    "parent_section": 0.92,
    "router_summary": 0.85,
}

# Long narrative passages accumulate query terms simply by being long, which lets
# them outrank the short recommendation that actually answers the question. Term
# coverage is therefore damped by length above this size.
LENGTH_NORM_CHARS = 600

# Weight on the user's literal query vs its synonym expansion.
W_ORIGINAL_QUERY = 0.7

# A number in the query falling inside a band the chunk states is a hard match, not
# a soft similarity - "38" inside "30 - 44" is the whole answer to "which band?".
# As one term inside the metadata signal it contributed ~0.035 after weighting and
# rerank blending, which lost to a 0.32 BM25 gap; the correct staging row ranked
# second behind a gadolinium-contrast practice point by 0.008. Applied as a prior it
# scales the whole relevance, in the same way the chunk-type prior does, and stays
# query-derived rather than keyed to any particular question.
# A number in the query falling inside a band the chunk states is a hard match, not a
# soft similarity: "38" inside "30 - 44" is the whole answer to "which band?". At 0.10
# the file said that and did not act on it - a perfect band match moved a chunk about
# 7%, halved again by the rerank blend, which is not enough to lift the row that answers
# the question over its sibling rows. The cross-encoder cannot help here: it is choosing
# between table rows that differ only in their numbers.
#
# Swept 0.10 / 0.25 / 0.40 / 0.60 on the gold set, the staging slice and the held-out
# set. 0.40 is best everywhere at once - staging 0.4629 -> 0.5546, held-out clinician
# 0.5305 -> 0.5492, gold set 0.7091 -> 0.7152. At 0.60 the prior starts overriding
# relevance rather than nudging it and the gold set falls to 0.6851.
NUMERIC_BAND_PRIOR = 0.40

# How much of that prior is earned by *selectivity* rather than mere membership.
#
# The prior above asked one question - "does this chunk state a range covering the
# patient's number?" - and answered it yes or no. For "filtration rate came back at 38",
# 7 of the 8 surviving candidates answered yes, so the signal separated nothing and the
# ranking fell entirely to the cross-encoder, which put KDIGO Table 24 ("variation of
# laboratory values ... by age group, sex, and eGFR") first and left Table 2, the
# staging table, outside the top 8.
#
# The two chunks are not equally responsive. Table 2's row states one band, 30-44, and
# 38 is in it: an unambiguous assignment. Table 24's matching row states seven bands -
# 90-104, 75-89, 60-74, 45-59, 30-44, ... - because it is the header enumerating the
# columns a stratified table is sliced by. It spans 38 without assigning it.
#
# So a share of the prior is paid for exclusivity: what fraction of the chunk's bands
# for this quantity actually contain the value. At 0.0 this reduces exactly to the old
# membership test, which is what makes it the honest baseline in the sweep.
#
# Swept 0.0 / 0.25 / 0.5 / 0.75 / 1.0 on the gold set, the held-out set and the clinical
# FAQ set (scripts/sweep_band_selectivity.py, reports/band_selectivity_sweep.json):
#
#     0.0   0.7075   0.8411   20/20      the shipped membership test
#     0.25  0.7119   0.8411   20/20      [+0.000, +0.013]
#     0.5   0.7250   0.8411   20/20      [+0.005, +0.033]  <- kept
#     0.75  0.7217   0.8411   20/20      [+0.001, +0.031]
#     1.0   0.7243   0.8401   20/20      [+0.004, +0.033], held-out CI tops out at 0.000
#
# 0.5, 0.75 and 1.0 are indistinguishable from each other on the gold set; 0.5 is taken
# as the least aggressive of them that costs nothing on the held-out set. The held-out
# set is flat across the middle three, which is a fact about that set rather than a
# vindication: it contains few questions where several chunks span the same value.
#
# It does not win the query that prompted it. For "filtration rate 38" the gold row goes
# from rank 16 to 10 while KDIGO Table 24 keeps first place, because the cross-encoder
# scores that chunk 1.000 and rerank is half the final score - this prior is applied
# before reranking and can only move who enters the window. The saturation is fixed; the
# cross-encoder's preference for a caption reading "variation of laboratory values ...
# by eGFR" is a separate, unfixed problem.
BAND_SELECTIVITY = 0.5

STOPWORDS = {
    "what", "when", "which", "who", "whom", "whose", "where", "why", "how", "does",
    "do", "did", "is", "are", "was", "were", "be", "been", "should", "would", "could",
    "can", "may", "might", "must", "the", "a", "an", "and", "or", "but", "if", "then",
    "for", "with", "without", "in", "on", "at", "to", "of", "from", "by", "about",
    "into", "as", "that", "this", "these", "those", "it", "its", "there", "their",
    "them", "they", "my", "our", "your", "patient", "person", "people", "adults",
    "adult", "say", "says", "said", "use", "used", "using", "guideline", "guidelines",
    "recommend", "recommended", "actually", "really", "need", "needs", "given", "get",
    # Ordinary English that carries no topical signal. These matter beyond ranking:
    # answer.py refuses a question as out-of-domain when a content term appears
    # nowhere in the corpus, and "something" is not in a clinical guideline, so
    # "how much of a rise actually means something?" was declined as though it had
    # asked about appendicitis. A generic word must not be able to look like an
    # unknown clinical concept.
    "something", "anything", "everything", "nothing", "someone", "anyone",
    "usually", "normally", "generally", "typically", "often", "sometimes",
    "much", "many", "more", "most", "less", "least", "long", "time", "times",
    "point", "thing", "things", "way", "ways", "make", "makes", "come", "comes",
    "look", "looks", "mean", "means", "meaning", "know", "think", "want",
    "good", "bad", "better", "worse", "best", "worst", "just", "still", "also",
}

REFERENCE_LIKE = re.compile(
    r"(?:\b\d{4};\s?\d+\s?[:(]|\bet al\b|\bdoi\b|\bPMID\b|\bCD0\d{5}\b"
    r"|\b(?:Lancet|JAMA|BMJ|PLoS|Kidney Int|N Engl J Med|Cochrane)\b)",
    re.I,
)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)?|[<>]=?|\bG[1-5][ab]?\b|\bA[1-3]\b", text.lower())


def content_terms(text: str) -> set[str]:
    """Query tokens carrying actual topical signal."""
    return {term for term in tokenize(text) if len(term) > 2 and term not in STOPWORDS}


# --------------------------------------------------------------------------- io

@dataclass
class Indexes:
    records: list[dict]
    record_by_id: dict[str, dict]
    tfidf: dict
    dense: np.ndarray | None
    dense_meta: dict


_INDEXES: Indexes | None = None

# Three states, not two. "Nobody chose" must resolve to the real embeddings; "someone
# chose None" must stay off, because ablate_embeddings.py disables dense on purpose to
# get its lexical-only reference point. Collapsing the two is what made the default
# silently lexical: the legacy path data/indexes/dense_embeddings.npy has not existed
# since embeddings moved into dense/<model>.npy, so an unset path found no file, set
# dense to None, and every caller that passed --dense-model without --dense-file ran
# without dense retrieval and was told nothing. That is worth 0.064 nDCG (0.7062 ->
# 0.6429), and it is the reason the first HyDE measurement moved nothing: there was no
# dense query vector to blend a hypothetical into.
_UNSET = object()
_DENSE_PATH: object = _UNSET


def set_dense_path(path) -> None:
    """Point retrieval at a specific embedding file, resetting the cache.

    Passing None disables dense retrieval deliberately.
    """
    global _DENSE_PATH, _INDEXES
    _DENSE_PATH = Path(path) if path else None
    _INDEXES = None


# The shipped configuration, named rather than inferred. Resolving "the only .npy in
# the directory" worked until a second encoder was built for the MedCPT ablation, at
# which point every documented command failed. Falling back to "whichever sorts first"
# would have been worse: an ablation artefact would silently have become the default.
DEFAULT_DENSE_FILE = "abhinand__MedEmbed-large-v0_1.npy"


def default_dense_path() -> Path | None:
    """The embeddings to use when no caller has chosen."""
    legacy = INDEX_DIR / "dense_embeddings.npy"
    if legacy.exists():
        return legacy
    shipped = INDEX_DIR / "dense" / DEFAULT_DENSE_FILE
    if shipped.exists():
        return shipped
    candidates = sorted((INDEX_DIR / "dense").glob("*.npy"))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise SystemExit(
            f"the default embeddings ({DEFAULT_DENSE_FILE}) are missing and "
            f"{len(candidates)} others are present, so there is no safe default. "
            "Pass --dense-file so the run records which vectors it used:\n  "
            + "\n  ".join(str(c) for c in candidates)
        )
    return None


def load_indexes() -> Indexes:
    """Load once per process. The evaluator calls retrieve() ~60 times."""
    global _INDEXES
    if _INDEXES is not None:
        return _INDEXES

    records = json.loads((INDEX_DIR / "records.json").read_text(encoding="utf-8"))
    with (INDEX_DIR / "tfidf.pkl").open("rb") as f:
        tfidf = pickle.load(f)

    dense_path = default_dense_path() if _DENSE_PATH is _UNSET else _DENSE_PATH
    dense = np.load(dense_path) if dense_path and dense_path.exists() else None
    dense_meta = {}
    if dense_path:
        meta_path = dense_path.with_suffix(".meta.json")
        if meta_path.exists():
            dense_meta = json.loads(meta_path.read_text(encoding="utf-8"))

    if dense is not None and len(dense) != len(records):
        raise SystemExit(
            f"dense embeddings hold {len(dense)} vectors but the corpus has {len(records)} chunks. "
            "Rebuild embeddings after re-chunking."
        )

    _INDEXES = Indexes(records, {r["id"]: r for r in records}, tfidf, dense, dense_meta)
    return _INDEXES


_DENSE_MODELS: dict[str, object] = {}
_RERANKERS: dict[str, object] = {}


def get_dense_model(name: str):
    if name not in _DENSE_MODELS:
        from nephrolex.retrieval.medcpt import MedCPTEncoder, is_medcpt

        if is_medcpt(name):
            # Queries go through MedCPT's query encoder, which is a different set of
            # weights from the one that embedded the corpus. That asymmetry is the
            # point of the model, not an accident to be smoothed over.
            _DENSE_MODELS[name] = MedCPTEncoder("query")
        else:
            from sentence_transformers import SentenceTransformer

            _DENSE_MODELS[name] = SentenceTransformer(name, trust_remote_code=True)
    return _DENSE_MODELS[name]


def get_reranker(name: str):
    if name not in _RERANKERS:
        from sentence_transformers import CrossEncoder

        _RERANKERS[name] = CrossEncoder(name)
    return _RERANKERS[name]


# ------------------------------------------------------------------- signals
#
# The relevance score is a fold over this registry rather than a hand-written
# expression, for one concrete reason: scripts/ablate_components.py used to restate
# the component list by hand, and a variant there was misconfigured to the same value
# as the baseline - producing a confident "0.0000, no effect" row for a component that
# had never actually been tested. Enumerating the registry means the ablation cannot
# drift from the thing it measures. Adding a signal here adds its ablation row.
#
# Additive signals are weighted and normalised, then divided by the total weight.
# Multiplicative ones are priors applied to the result. Order is load-bearing only in
# that it must not change: the fold reproduces the original expression bit-for-bit.

QUALITY_PENALTY_ON = True


@dataclass(frozen=True)
class Signal:
    """One contribution to the fused score, and how to switch it off."""

    name: str
    additive: bool
    weight_param: str | None      # retrieve() keyword carrying this signal's weight
    disable: dict                 # module constants to override in order to disable it
    note: str


SIGNALS: tuple[Signal, ...] = (
    Signal("tfidf", True, "w_tfidf", {},
           "lexical, length-normalised; carries exact tokens now that BM25 is out"),
    Signal("dense", True, "w_dense", {},
           "MedEmbed-large bi-encoder cosine; the paraphrase signal"),
    Signal("meta", True, "w_meta", {},
           "query-vocabulary coverage of the chunk, plus exact and numeric hits"),
    Signal("colbert", True, "w_colbert", {},
           "late-interaction MaxSim; 0.0 by default - redundant with the cross-encoder"),
    Signal("doc2query", True, "w_doc2query", {},
           "lexical match against generated clinician-voice questions, indexed separately "
           "so generated text never reaches a citation"),
    Signal("chunk_type_prior", False, None, {"CHUNK_TYPE_PRIOR": {}},
           "recommendations and table rows over prose; the second-largest contributor"),
    Signal("numeric_band", False, None, {"NUMERIC_BAND_PRIOR": 0.0},
           "quantity-aware range match, so 'filtration rate 38' finds the GFR band"),
    Signal("quality_penalty", False, None, {"QUALITY_PENALTY_ON": False},
           "down-weights bibliography-like text and non-citable chunks"),
)


# ------------------------------------------------------------- document expansion

_DOC2QUERY: dict | None = None


def doc2query_text(index: int) -> str:
    """The generated questions for one chunk, for showing to the cross-encoder."""
    if _DOC2QUERY is None or not _DOC2QUERY.get("ok"):
        return ""
    texts = _DOC2QUERY.get("texts") or []
    return texts[index] if index < len(texts) else ""


def doc2query_scores(query: str) -> np.ndarray | None:
    """Lexical score against each chunk's generated questions, or None if not built.

    Kept in its own index rather than appended to the chunk text. Concatenating
    expansions into the passage is reported to hurt dense retrieval, and here it would
    also put generated sentences inside the text the answer layer quotes from. A
    hallucinated question can cost ranking; it must never be able to reach a citation.
    """
    global _DOC2QUERY
    if _DOC2QUERY is None:
        path = INDEX_DIR / "doc2query_tfidf.pkl"
        if not path.exists():
            _DOC2QUERY = {"ok": False}
        else:
            with path.open("rb") as f:
                index = pickle.load(f)
            # Same staleness guard as the ColBERT index: a rebuilt corpus leaves this
            # a different length, and unchecked it hands back candidate indices past
            # the end of the corpus for the other retrievers to crash on.
            if index["matrix"].shape[0] != len(load_indexes().records):
                print(
                    f"WARNING: doc2query index has {index['matrix'].shape[0]} rows but the "
                    f"corpus has {len(load_indexes().records)}; ignoring it. Rebuild with "
                    "scripts/build_doc2query_index.py",
                    file=sys.stderr,
                )
                _DOC2QUERY = {"ok": False}
            else:
                _DOC2QUERY = {"ok": True, **index}

    if not _DOC2QUERY.get("ok"):
        return None
    vector = _DOC2QUERY["vectorizer"].transform([query])
    return (_DOC2QUERY["matrix"] @ vector.T).toarray().ravel()


# --------------------------------------------------------------- late interaction

_COLBERT: dict | None = None


def colbert_scores(query: str) -> np.ndarray | None:
    """Per-chunk ColBERT MaxSim, or None if the token index has not been built.

    Loaded lazily and only when w_colbert > 0, so the default path costs nothing.
    Token vectors are ~35 MB for this corpus, and a query scores in ~55 ms, which
    is why the whole corpus is scored exhaustively instead of using an ANN stage.
    """
    global _COLBERT
    if _COLBERT is None:
        from nephrolex.retrieval import colbert as _cb

        if not (_cb.OUT_DIR / "tokens.npy").exists():
            _COLBERT = {"ok": False}
        else:
            docs, records = _cb.load_index()
            # The token index is built from a snapshot of the corpus, so a rebuild
            # leaves it a different length. Unchecked, it returns candidate indices
            # past the end of the corpus and the *other* retrievers crash on them -
            # "index 1505 is out of bounds for axis 0 with size 1479", raised inside
            # BM25 normalisation, pointing nowhere near the stale index that caused it.
            # The dense path has had this guard from the start; this one did not.
            corpus_size = len(load_indexes().records)
            if len(docs) != corpus_size:
                _COLBERT = {"ok": False}
                print(
                    f"colbert index holds {len(docs)} chunks but the corpus has "
                    f"{corpus_size}; ignoring it. Rebuild with: "
                    "python scripts/colbert.py build",
                    file=sys.stderr,
                )
            else:
                tokenizer, model, projection, device = _cb.load_model()
                _COLBERT = {"ok": True, "module": _cb, "docs": docs, "tokenizer": tokenizer,
                            "model": model, "projection": projection, "device": device,
                            "position": {r["id"]: i for i, r in enumerate(records)}}
    if not _COLBERT["ok"]:
        return None
    cb = _COLBERT
    return cb["module"].score_all(query, cb["docs"], cb["tokenizer"], cb["model"],
                                  cb["projection"], cb["device"])


# ---------------------------------------------------------------------- scoring

def expand_query(query: str) -> str:
    lower = query.lower()
    expansions: list[str] = []
    for trigger, terms in QUERY_EXPANSIONS.items():
        if trigger in lower:
            expansions.extend(terms)
    if re.search(r"\bG[1-5][ab]?\b|\bA[1-3]\b", query, re.I):
        expansions.extend(["CGA", "eGFR category", "albuminuria category", "classification of CKD"])
    if not expansions:
        return query
    deduped = list(dict.fromkeys(term for term in expansions if term.lower() not in lower))
    return f"{query} {' '.join(deduped)}"


def normalize(scores: np.ndarray, indices: list[int]) -> dict[int, float]:
    """Min-max normalise into [0, 1] over the candidate pool only."""
    if not indices:
        return {}
    values = scores[indices]
    low, high = float(values.min()), float(values.max())
    if high - low < 1e-12:
        return {idx: 0.0 for idx in indices}
    return {idx: (float(scores[idx]) - low) / (high - low) for idx in indices}


def top_indices(scores: np.ndarray, n: int) -> list[int]:
    if scores.size == 0:
        return []
    n = min(n, scores.size)
    partition = np.argpartition(scores, -n)[-n:]
    return [int(idx) for idx in partition[np.argsort(scores[partition])[::-1]]]


RANGE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:-|–|to)\s*(\d+(?:\.\d+)?)")
# The Unicode comparators are the ones the guidelines actually use - "≥ 90" for GFR
# category G1, "< 15" for G5 - and listing only their ASCII spellings meant the two
# open-ended categories, the ones a range cannot express, had no band at all. So
# "their filtration number is 95, is that in the normal band?" scored 0.000 against
# the row that answers it.
BOUND_RE = re.compile(
    r"(<=|>=|≤|≥|<|>|less than|more than|at least|no less than|under|over|below|above)"
    r"\s*(\d+(?:\.\d+)?)",
    re.I,
)

# What quantity a query is talking about, and how that quantity is written in the
# guidelines. Without this, "filtration rate 38" matched any band spanning 38 -
# including one inside an abbreviation footnote about metabolic acidosis, which then
# outranked the GFR staging table.
QUANTITY_CONTEXT = {
    "egfr": r"e?GFR|glomerular filtration|ml\s?/\s?min",
    "acr": r"\bACR\b|albumin|mg\s?/\s?mmol|mg\s?/\s?g",
    "bp": r"systolic|diastolic|blood pressure|mm\s?Hg",
    "potassium": r"potassium|mmol\s?/\s?l",
    "haemoglobin": r"h(?:a)?emoglobin|\bHb\b|g\s?/\s?l",
}

QUERY_QUANTITY = [
    ("egfr", r"e?GFR|glomerular filtration|filtration rate|kidney function|creatinine clearance"),
    ("acr", r"\bACR\b|albumin|albuminuria|proteinuria"),
    ("bp", r"blood pressure|systolic|diastolic"),
    ("potassium", r"potassium|hyperkal"),
    ("haemoglobin", r"h(?:a)?emoglobin|\bHb\b|an(?:a)?emia"),
]

CONTEXT_WINDOW = 70


def query_quantity(query: str) -> str | None:
    for name, pattern in QUERY_QUANTITY:
        if re.search(pattern, query, re.I):
            return name
    return None


# Ages and durations get spelled out in speech and in anything transcribed from it -
# "a child of eleven", "a man in his sixties", "for eighteen months". The band matcher
# reads digits, so a spelled number left it with nothing to anchor on, and nothing to
# anchor on is not a neutral outcome: asked how much salt is right for a child of
# eleven, the system ranked KDIGO Table 22's *infant* row - 0 to 6 months, 0.110 g/day -
# above the 9-to-13-year row that answers it. Right table, wrong row, real citation, and
# a tenfold error in a paediatric dose. With "11" in place of "eleven" the same signal
# scores the correct row 1.0 and the infant row 0.0.
_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19,
}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
         "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}

_TENS_ALT = "|".join(sorted(_TENS, key=len, reverse=True))
_ONES_ALT = "|".join(sorted((k for k in _UNITS if 0 < _UNITS[k] < 10), key=len, reverse=True))
_UNITS_ALT = "|".join(sorted(_UNITS, key=len, reverse=True))

# "sixty-five" and "sixty five" before bare "sixty", and any spelled unit otherwise.
# Word boundaries matter: without them "one" fires inside "money" and "ten" inside
# "often", feeding junk into a signal meant to be a hard match.
_SPELLED = re.compile(
    rf"\b(?:({_TENS_ALT})[\s-]?({_ONES_ALT})?|({_UNITS_ALT}))\b",
    re.I,
)


def spelled_numbers(text: str) -> list[float]:
    """Numbers written as words, so "eleven" behaves exactly like "11"."""
    found: list[float] = []
    for tens, unit, plain in _SPELLED.findall(text):
        if tens:
            found.append(float(_TENS[tens.lower()] + (_UNITS[unit.lower()] if unit else 0)))
        elif plain:
            found.append(float(_UNITS[plain.lower()]))
    return found


def numeric_range_signal(query: str, raw: str) -> float:
    """Does a number in the query fall inside a band the chunk states *for the same
    quantity*?

    Guidelines are written as bands - GFR 30-44 is G3b - but a clinician asks "my
    patient's eGFR is 38, which category?". No lexical or embedding match connects 38
    to "30-44", because the token 38 never appears.

    The quantity check is what makes this usable rather than noisy: a bare
    band-containment test fires on any number in range anywhere in a chunk, so the
    signal has to be anchored to a mention of the same measurement nearby.
    """
    values = [float(v) for v in re.findall(r"\b(\d+(?:\.\d+)?)\b", query)]
    values += spelled_numbers(query)
    if not values:
        return 0.0

    quantity = query_quantity(query)
    context = QUANTITY_CONTEXT.get(quantity) if quantity else None

    def has_context(start: int, end: int) -> bool:
        if not context:
            return True
        window = raw[max(0, start - CONTEXT_WINDOW) : end + CONTEXT_WINDOW]
        return bool(re.search(context, window, re.I))

    bands: list[tuple[float, float]] = []
    for match in RANGE_RE.finditer(raw):
        if has_context(match.start(), match.end()):
            bands.append((float(match.group(1)), float(match.group(2))))
    for match in BOUND_RE.finditer(raw):
        if not has_context(match.start(), match.end()):
            continue
        bound = float(match.group(2))
        operator = match.group(1).lower()
        if operator in {"<", "<=", "less than", "under", "below"}:
            bands.append((0.0, bound))
        else:
            bands.append((bound, float("inf")))

    if not bands:
        return 0.0

    # Membership earns (1 - BAND_SELECTIVITY); the rest is earned by how exclusively
    # the chunk's bands point at this value. One band containing it scores full marks;
    # one of seven scores the membership share plus a seventh of the remainder.
    total = 0.0
    for value in values:
        containing = sum(1 for low, high in bands if low <= value <= high)
        if not containing:
            continue
        total += (1.0 - BAND_SELECTIVITY) + BAND_SELECTIVITY * (containing / len(bands))
    return min(total / len(values), 1.0)


def metadata_signal(query: str, record: dict) -> float:
    """Query-independent lexical grounding signal in [0, 1].

    Three components, all derived from the query itself rather than from any
    hand-written per-question rule:
      - coverage: how much of the query's content vocabulary appears in the chunk
      - exact:    overlap with the chunk's extracted clinical terms/thresholds
      - numeric:  when the query carries numbers or units, does the chunk too
    """
    terms = content_terms(query)
    if not terms:
        return 0.0

    raw = record["raw_text"].lower()
    coverage = sum(1 for term in terms if term in raw) / len(terms)
    if len(raw) > LENGTH_NORM_CHARS:
        coverage *= (LENGTH_NORM_CHARS / len(raw)) ** 0.5

    exact_terms = {term.lower() for term in record["metadata"].get("exact_terms", [])}
    exact = 0.0
    if exact_terms:
        matched = sum(1 for term in exact_terms if term in query.lower())
        exact = min(matched / 3.0, 1.0)

    query_numbers = set(re.findall(r"\d+(?:\.\d+)?", query))
    numeric = 0.0
    if query_numbers:
        chunk_numbers = set(re.findall(r"\d+(?:\.\d+)?", raw))
        numeric = min(len(query_numbers & chunk_numbers) / len(query_numbers), 1.0)

    in_band = numeric_range_signal(query, raw)

    return 0.40 * coverage + 0.15 * exact + 0.10 * numeric + 0.35 * in_band


def quality_penalty(record: dict) -> float:
    """Down-weight text that reads like a bibliography rather than guidance."""
    penalty = 1.0
    if REFERENCE_LIKE.search(record["raw_text"]):
        penalty *= 0.55
    section = " ".join(record["metadata"].get("section_path", [])).lower()
    if "reference" in section:
        penalty *= 0.65
    if not record["metadata"].get("citable", True):
        penalty *= 0.90
    return penalty


# ------------------------------------------------------------------- selection

def section_key(record: dict) -> tuple[str, str]:
    metadata = record["metadata"]
    return metadata.get("document_id", ""), " > ".join(metadata.get("section_path", []))


def _fingerprint(record: dict) -> str:
    return re.sub(r"[^a-z0-9]", "", record["raw_text"].lower())[:160]


# The value-and-unit pairs a chunk actually states. Used to tell a sibling that repeats
# a section from one that completes it.
# Off restores the plain section penalty, so the exemption can be measured rather than
# assumed. Enumerated by the ablation like every other switch in this file.
COMPLEMENTARY_EXEMPTION = True

WHITESPACE = re.compile(r"\s+")

STATED_VALUE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(mm\s?Hg|mg\s?/\s?mmol|mg\s?/\s?g|ml\s?/\s?min|mmol\s?/\s?l|"
    r"g\s?/\s?kg|g\s?/\s?l|g\s?/\s?d|%)",
    re.I,
)


def stated_values(record: dict) -> frozenset:
    """Thresholds this chunk states, normalised so "120 mm Hg" and "120mmHg" agree."""
    found = set()
    for value, unit in STATED_VALUE.findall(record["raw_text"]):
        found.add(value + WHITESPACE.sub("", unit).lower())
    return frozenset(found)


def diversify(candidates: list[int], records: list[dict], scores: dict[int, float], top_k: int,
              dedupe: bool = False) -> list[int]:
    """Greedy MMR-style selection penalising repeats from the same section.

    `dedupe` additionally drops a chunk whose text has already been selected. KDIGO
    prints every recommendation twice - once in its summary and again in the chapter -
    which the section penalty above cannot catch, because the two copies sit in
    different sections. 23% of the corpus is an exact duplicate of another chunk, and
    duplicates occupy 9% of top-10 slots.

    Off by default: the answer layer already removes restatements before quoting
    (answer._distinct), so this only changes what retrieval itself returns, and
    whether that helps is measured rather than assumed.
    """
    if top_k <= 0:
        return []
    remaining = candidates.copy()
    selected: list[int] = []
    section_counts: dict[tuple[str, str], int] = {}
    section_values: dict[tuple[str, str], set] = {}
    seen_text: set[str] = set()
    values = {idx: stated_values(records[idx]) for idx in candidates}

    while remaining and len(selected) < top_k:
        best_idx, best_score = None, -math.inf
        for idx in remaining:
            adjusted = scores.get(idx, 0.0)
            key = section_key(records[idx])
            seen = section_counts.get(key, 0)
            if seen:
                # Two chunks from one section are usually near-duplicates, which is what
                # the penalty is for. They are not, when each states a threshold the
                # other does not: NICE 1.6.1 and 1.6.2 sit in "Blood pressure control"
                # and give the two arms of one decision - below 140 when ACR is under
                # 70, below 130 when it is 70 or more. Penalising the second as a repeat
                # dropped it from rank 4 to rank 8 and out of the citable evidence, so a
                # correct answer naming both arms could not be grounded and was
                # discarded by the verifier.
                #
                # A chunk offering a value none of its selected siblings offer is
                # completing the section, not repeating it, and is left alone.
                novel = values[idx] - section_values.get(key, set())
                if not (novel and COMPLEMENTARY_EXEMPTION):
                    adjusted *= 0.80**seen
            if adjusted > best_score:
                best_idx, best_score = idx, adjusted
        if best_idx is None:
            break
        remaining.remove(best_idx)
        if dedupe:
            text = _fingerprint(records[best_idx])
            if text in seen_text:
                continue
            seen_text.add(text)
        selected.append(best_idx)
        key = section_key(records[best_idx])
        section_counts[key] = section_counts.get(key, 0) + 1
        section_values.setdefault(key, set()).update(values.get(best_idx, ()))
    return selected




def attach_parent_context(result: dict, record_by_id: dict[str, dict]) -> dict:
    parent_id = result["metadata"].get("parent_id")
    if not parent_id or result["metadata"].get("chunk_type") == "parent_section":
        return result
    parent = record_by_id.get(parent_id)
    if parent:
        result["parent_context"] = {
            "id": parent["id"],
            "raw_text": parent["raw_text"][:1800],
            "metadata": parent["metadata"],
        }
    return result


# --------------------------------------------------------------------- pipeline

def retrieve(
    query: str,
    top_k: int,
    dense_model: str | None = None,
    reranker_model: str | None = None,
    *,
    w_tfidf: float = W_TFIDF,
    w_dense: float = W_DENSE,
    w_meta: float = W_META,
    w_colbert: float = 0.0,
    # Lexical weight on the generated clinician-voice questions. 0.0 skips the index
    # load entirely, so the default path pays nothing for a feature it is not using.
    #
    # 0.4 measured. On the held-out chunk-first set this is +0.1612 nDCG@10 on
    # clinician-phrased questions, 95% CI [+0.0747, +0.2581], against +0.0033 on
    # echo-phrased ones - it lifts the weak side without costing the strong one. On the
    # 67-question gold set it is -0.0037, inside the noise floor, while total misses fall
    # from 5 to 3 and recall@20 rises. It is the only technique tried here that cleared
    # its confidence interval; Contextual Retrieval, ColBERT, HyDE and MedCPT did not.
    w_doc2query: float = W_DOC2QUERY,
    w_band: float = W_BAND,
    doc2query_mode: str | None = None,
    # Whether the cross-encoder is shown those questions as well as the chunk.
    #
    # On. The reranker is half the final score and, reading raw text alone, it is blind
    # to the signal that surfaced the candidate. Asked "how much does ACR have to rise
    # before it means something?", the doc2query signal ranks KDIGO Practice Point 2.1.5
    # first on its own; the reranker then fills the top four with near-identical rows of
    # NICE's monitoring table and pushes the answer out of the top twenty entirely.
    #
    # This was shipped, measured, reverted, and reinstated when a third benchmark
    # arrived. The evidence is genuinely mixed and is recorded in full rather than
    # summarised to the favourable half:
    #
    #   clinician-FAQ set (20 questions, externally sourced from the primary-care
    #     literature): success@5 90% -> 100%, nDCG 0.5456 -> 0.7013. Both failures fixed.
    #   held-out chunk-first set (46): +0.1360 nDCG, 95% CI [+0.0538, +0.2299], with the
    #     clinician-phrased slice going 0.5492 -> 0.7625.
    #   67-question gold set: -0.0072 nDCG, inside the noise floor, but recall@20 falls
    #     0.8483 -> 0.8184 and total misses rise 2 -> 5. End-to-end audit 69/75 -> 67/75.
    #
    # What decided it: the three extra misses are mostly cases already failing at 0.185
    # to 0.316 tipping below zero, against one genuine 1.000 -> 0 (`mon_aki_paraphrase`)
    # and one 0 -> 1.000 rescue (`mon_acr_paraphrase`). Two benchmarks that do not share
    # an origin with this feature favour it; the one that does not is neutral on its
    # headline metric.
    #
    # The honest caveat: `egfr_confirm` and `faq_04` are the same clinical question in
    # different words, and this fixes one while breaking the other. It trades phrasing
    # sensitivity rather than removing it.
    rerank_with_doc2query: bool = True,
    w_rerank: float = 1.0,
    rerank_depth: int = 30,
    dedupe: bool = False,
    # How the hypothetical passage is used: "off" costs nothing (no cache read, no
    # model call), "max" takes the better of question-match and answer-match per
    # chunk, "blend" moves the query vector itself. See the block that reads this.
    hyde_mode: str = "off",
    # Only read when hyde_mode == "blend".
    w_hyde: float = 0.0,
    # Whether a cache miss may call the LLM. Off during evaluation so a run is
    # reproducible and free; on in the live system where latency is the only cost.
    hyde_generate: bool = False,
    # Which retrievers may nominate candidates. Defaults to all of them; narrowing it
    # is how a component is genuinely removed rather than merely silenced in scoring.
    pool_from: tuple[str, ...] = ("tfidf", "dense", "colbert", "doc2query"),
    fusion: str = "convex",
    explain: bool = False,
) -> list[dict]:
    indexes = load_indexes()
    records = indexes.records
    expanded = expand_query(query)

    # --- per-retriever raw scores -----------------------------------------
    # The original query and its expansion are scored separately and blended, with
    # the original dominant. Scoring only the expanded string lets a dozen synonym
    # terms outvote what the user actually typed: for "GFR category G3b" the
    # expansion adds CGA, classification, albuminuria category and more, which
    # favours long passages mentioning all of them over the table row that states
    # G3b = 30-44.
    def blended(score_fn) -> np.ndarray:
        original = np.asarray(score_fn(query))
        if expanded == query:
            return original
        return W_ORIGINAL_QUERY * original + (1 - W_ORIGINAL_QUERY) * np.asarray(score_fn(expanded))

    tfidf_scores = blended(
        lambda text: (indexes.tfidf["matrix"] @ indexes.tfidf["vectorizer"].transform([text]).T)
        .toarray()
        .ravel()
    )

    dense_scores = None
    if dense_model and indexes.dense is None:
        # Asking for a dense model and getting lexical-only retrieval without a word
        # said is how a whole component goes missing from a measurement while the
        # report still names the model. Refuse instead.
        raise SystemExit(
            f"dense model {dense_model} was requested but no embeddings are loaded. "
            f"Expected a .npy in {INDEX_DIR / 'dense'} - run scripts/build_indexes.py, "
            "or pass --dense-file, or drop --dense-model to run lexical-only on purpose."
        )
    if dense_model and indexes.dense is not None:
        built_with = indexes.dense_meta.get("model_name")
        if built_with and built_with != dense_model:
            raise SystemExit(
                f"embeddings were built with {built_with} but the query model is {dense_model}. "
                "Vectors from different models are not comparable; rebuild or pass the matching model."
            )
        model = get_dense_model(dense_model)
        prefix = indexes.dense_meta.get("query_prefix", "")
        query_vec = model.encode([prefix + query], normalize_embeddings=True)[0]

        dense_scores = indexes.dense @ query_vec

        # HyDE: embed the passage a guideline would contain if it answered this
        # question, and search with that too. It is a search key only - nothing from
        # it reaches the answer, the citation or the verifier - so a wrong
        # hypothetical can cost ranking but never grounding.
        #
        # Two ways to use it, and the difference is not cosmetic:
        #
        #   blend  moves the single query vector toward the hypothetical. This is the
        #          published formulation and it is destructive: the vector leaves the
        #          neighbourhood the literal question pointed at. Measured here, a
        #          correct hypothetical written in KDIGO's register pushed the NICE
        #          gold chunk for def_paraphrase out of the top 20 entirely (recall@20
        #          1.00 -> 0.00), cancelling three total-miss rescues elsewhere.
        #
        #   max    keeps both vectors and takes the better match per chunk: does this
        #          chunk look like the question, or like the answer the question
        #          expects? Measured better than blend on nDCG, recall@20 and MRR, and
        #          it has no weight to fit.
        #
        # A chunk's dense score can only rise under max, but that does not protect its
        # rank, and assuming it did was wrong: for gfr_g5 the hypothetical was generic
        # enough to lift 66 other chunks past the gold one. The subtler case is
        # def_paraphrase, where max *improved* the gold chunk's dense rank 18 -> 3 and
        # the final answer still lost it - the cross-encoder scores an unchanged
        # (query, chunk) pair, so what changed was which competitors entered its
        # window. Widening recall hands the reranker new candidates it may prefer.
        #
        # Both are normalised cosine similarities against the same embedding matrix,
        # so a max between them is a comparison of like with like.
        if hyde_mode != "off":
            from nephrolex.retrieval.hyde import hypothetical

            guess = hypothetical(query, allow_generate=hyde_generate)
            if guess:
                guess_vec = model.encode([prefix + guess], normalize_embeddings=True)[0]
                if hyde_mode == "max":
                    dense_scores = np.maximum(dense_scores, indexes.dense @ guess_vec)
                elif hyde_mode == "blend":
                    blended_vec = (1 - w_hyde) * query_vec + w_hyde * guess_vec
                    norm = np.linalg.norm(blended_vec)
                    if norm:
                        blended_vec = blended_vec / norm
                    dense_scores = indexes.dense @ blended_vec
                else:
                    raise SystemExit(f"unknown hyde_mode {hyde_mode!r}: use off, max or blend")

    # Late interaction is scored on the original query only: MaxSim already matches
    # each query token independently, so appending expansion synonyms adds tokens
    # that each demand their own best match and dilute the sum.
    colbert_raw = colbert_scores(query) if w_colbert > 0 else None
    # Scored on the original query only. The synonym expansion exists to bridge the
    # same vocabulary gap these questions were generated to bridge, and applying both
    # doubles the bridge and drifts the match.
    # Which query text this signal sees; see DOC2QUERY_QUERY_MODE for the measurement
    # behind the default. Scoring it on the expanded query is the intuitive choice, and
    # it does rescue the staging case, but it moved 15 of 67 gold cases and cost
    # gfr_initial_assessment half its score, so the default stays on the raw query.
    mode = DOC2QUERY_QUERY_MODE if doc2query_mode is None else doc2query_mode
    if w_doc2query <= 0:
        doc2query_raw = None
    elif mode == "raw":
        doc2query_raw = doc2query_scores(query)
    elif mode == "expanded":
        doc2query_raw = doc2query_scores(expanded)
    else:
        a, b = doc2query_scores(query), doc2query_scores(expanded)
        doc2query_raw = a if b is None else (b if a is None else np.maximum(a, b))

    # --- shared candidate pool --------------------------------------------
    # Every retriever nominates candidates regardless of its scoring weight, which is
    # deliberate - BM25 scores nothing and still earns its place by recall. It also
    # means setting a weight to zero does NOT remove that retriever: it keeps supplying
    # the pool, and the cross-encoder still reranks the top 30 of it. Leave-one-out
    # ablation by weight therefore measures "this signal's vote on ordering", not "this
    # component", and understates every component it names. `pool_from` exists so a
    # true removal can be measured rather than approximated.
    pool: list[int] = []
    for name, scores in (("tfidf", tfidf_scores),
                         ("dense", dense_scores), ("colbert", colbert_raw),
                         ("doc2query", doc2query_raw)):
        if scores is not None and name in pool_from:
            pool.extend(top_indices(scores, CANDIDATE_POOL))
    pool = list(dict.fromkeys(pool))
    if not pool:
        return []

    # --- normalised weighted combination ----------------------------------
    tfidf_norm = normalize(tfidf_scores, pool)
    dense_norm = normalize(dense_scores, pool) if dense_scores is not None else {}
    colbert_norm = normalize(colbert_raw, pool) if colbert_raw is not None else {}
    doc2query_norm = normalize(doc2query_raw, pool) if doc2query_raw is not None else {}

    meta_raw = {idx: metadata_signal(expanded, records[idx]) for idx in pool}
    band_match = {idx: numeric_range_signal(query, records[idx]["raw_text"]) for idx in pool}

    active_dense = w_dense if dense_norm else 0.0
    active_colbert = w_colbert if colbert_norm else 0.0
    active_doc2query = w_doc2query if doc2query_norm else 0.0
    total_weight = (w_tfidf + active_dense + w_meta + active_colbert
                    + active_doc2query + w_band)

    # Reciprocal Rank Fusion, for comparison against the convex combination used by
    # default. RRF is the more common choice in the literature; whether it is better
    # here is measured in scripts/ablate_components.py rather than assumed.
    rrf_rank: dict[int, float] = {}
    if fusion == "rrf":
        K = 60
        for scores in (tfidf_scores, dense_scores):
            if scores is None:
                continue
            for rank, idx in enumerate(top_indices(scores, CANDIDATE_POOL), start=1):
                rrf_rank[idx] = rrf_rank.get(idx, 0.0) + 1.0 / (K + rank)
        if rrf_rank:
            peak = max(rrf_rank.values())
            rrf_rank = {i: v / peak for i, v in rrf_rank.items()}

    # The registry, bound to this query's values. Insertion order matches the original
    # expression so the fold is bit-identical, not merely equivalent.
    additive: dict[str, tuple[float, dict]] = {
        "tfidf": (w_tfidf, tfidf_norm),
        "dense": (active_dense, dense_norm),
        "meta": (w_meta, meta_raw),
        "colbert": (active_colbert, colbert_norm),
        "doc2query": (active_doc2query, doc2query_norm),
        "band": (w_band, band_match),
    }

    def multiplicative(idx: int) -> dict[str, float]:
        return {
            "chunk_type_prior": CHUNK_TYPE_PRIOR.get(
                records[idx]["metadata"].get("chunk_type", ""), 1.0),
            "numeric_band": 1.0 + NUMERIC_BAND_PRIOR * band_match.get(idx, 0.0),
            "quality_penalty": quality_penalty(records[idx]) if QUALITY_PENALTY_ON else 1.0,
        }

    fused: dict[int, float] = {}
    components: dict[int, dict] = {}
    for idx in pool:
        if fusion == "rrf":
            relevance = rrf_rank.get(idx, 0.0)
        else:
            relevance = sum(
                weight * values.get(idx, 0.0) for weight, values in additive.values()
            ) / total_weight
        priors = multiplicative(idx)
        score = relevance
        for value in priors.values():
            score *= value
        fused[idx] = score
        if explain:
            components[idx] = {
                name: round(values.get(idx, 0.0), 4) for name, (_, values) in additive.items()
            }
            components[idx].update({name: round(value, 4) for name, value in priors.items()})
            components[idx]["numeric_band_match"] = round(band_match.get(idx, 0.0), 4)

    candidates = sorted(fused, key=lambda idx: fused[idx], reverse=True)[:CANDIDATE_POOL]

    # --- cross-encoder reranking ------------------------------------------
    if reranker_model:
        head = candidates[:rerank_depth]
        reranker = get_reranker(reranker_model)
        # Score the source text, not the templated retrieval text: the template
        # header is identical across chunks and would consume the cross-encoder's
        # limited token budget without carrying signal.
        # Optionally append the chunk's generated clinician-voice questions. The
        # cross-encoder is half the final score and, judging raw text alone, it is blind
        # to the one signal that surfaced the candidate: for "the urine protein result
        # has gone up - how much of a rise means something?", the doc2query signal ranks
        # KDIGO Practice Point 2.1.5 first on its own, and the reranker then buries it,
        # because the practice point says "a doubling of the ACR exceeds laboratory
        # variability" and shares almost no vocabulary with the question.
        #
        # This stays a ranking-only use of generated text. Nothing here reaches the
        # answer, the quote, the citation or the verifier - those all read raw_text.
        def rerank_text(idx: int) -> str:
            raw = records[idx]["raw_text"]
            if not rerank_with_doc2query:
                return raw
            expansion = doc2query_text(idx)
            return raw + "\n" + expansion if expansion else raw

        raw_scores = reranker.predict([(query, rerank_text(idx)) for idx in head])
        rerank_arr = np.asarray(raw_scores, dtype=float)
        rerank_norm = normalize(rerank_arr, list(range(len(head))))
        for position, idx in enumerate(head):
            blended = (fused[idx] + w_rerank * rerank_norm.get(position, 0.0)) / (1 + w_rerank)
            fused[idx] = blended
            if explain:
                components[idx]["rerank"] = round(rerank_norm.get(position, 0.0), 4)
        candidates = sorted(candidates, key=lambda idx: fused[idx], reverse=True)

    # --- selection ---------------------------------------------------------
    selected = diversify(candidates, records, fused, top_k, dedupe=dedupe)

    results = []
    for idx in selected:
        record = records[idx]
        result = {
            "score": round(fused.get(idx, 0.0), 6),
            "id": record["id"],
            "raw_text": record["raw_text"],
            "metadata": record["metadata"],
        }
        if explain:
            result["components"] = components.get(idx, {})
            result["expanded_query"] = expanded
        results.append(attach_parent_context(result, indexes.record_by_id))
    return results


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Hybrid CKD guideline retrieval.")
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--dense-model", default=None, help="Example: BAAI/bge-m3")
    parser.add_argument("--reranker-model", default=None, help="Example: BAAI/bge-reranker-v2-m3")
    # Every weight defaults to None so retrieve()'s signature stays the single source
    # of truth. Restating the numbers here would mean a change to the function silently
    # kept the old values on this path - the failure this module already hit once.
    parser.add_argument("--w-tfidf", type=float, default=None)
    parser.add_argument("--w-dense", type=float, default=None)
    parser.add_argument("--w-meta", type=float, default=None)
    parser.add_argument("--w-colbert", type=float, default=None)
    parser.add_argument("--w-rerank", type=float, default=None)
    parser.add_argument("--rerank-depth", type=int, default=None,
                        help="Measured better AND faster than 50: 0.6873 vs 0.6819.")
    parser.add_argument("--fusion", default="convex", choices=["convex", "rrf"])
    parser.add_argument("--explain", action="store_true", help="Include per-signal score breakdown.")
    args = parser.parse_args()

    overrides = {
        name: getattr(args, name)
        for name in ("w_tfidf", "w_dense", "w_meta", "w_colbert", "w_doc2query",
                     "w_rerank", "rerank_depth")
        if getattr(args, name) is not None
    }
    results = retrieve(
        args.query,
        args.top_k,
        args.dense_model,
        args.reranker_model,
        fusion=args.fusion,
        explain=args.explain,
        **overrides,
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
