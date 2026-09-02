"""Registry of embedding models for the retrieval ablation.

Prefixes matter and are easy to get wrong. Several of these models were trained
with asymmetric instructions - the query is prefixed, the passage is not, or both
are prefixed differently - and a model evaluated without its prefix scores well
below its real capability. Comparing a prefixed model against an unprefixed one
therefore measures the harness, not the models. Each entry records exactly what
the model expects, the prefix is applied at build time for passages and at query
time for queries, and the choice is written into the embeddings' sidecar metadata
so retrieval can never apply the wrong one.

Sources for the prefix conventions are the model cards on the HuggingFace Hub.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingModel:
    name: str
    params_m: int
    dims: int
    query_prefix: str = ""
    passage_prefix: str = ""
    note: str = ""

    @property
    def short(self) -> str:
        return self.name.split("/")[-1]


# The instruction BGE v1.5 English models were trained with for retrieval queries.
BGE_QUERY = "Represent this sentence for searching relevant passages: "

REGISTRY: dict[str, EmbeddingModel] = {
    model.name: model
    for model in [
        EmbeddingModel(
            "BAAI/bge-small-en-v1.5", 33, 384, query_prefix=BGE_QUERY,
            note="Smallest credible baseline; fast enough for CPU fallback.",
        ),
        EmbeddingModel(
            "BAAI/bge-base-en-v1.5", 109, 768, query_prefix=BGE_QUERY,
            note="Mid-size English retriever.",
        ),
        EmbeddingModel(
            "BAAI/bge-large-en-v1.5", 335, 1024, query_prefix=BGE_QUERY,
            note="Large English retriever.",
        ),
        EmbeddingModel(
            "BAAI/bge-m3", 568, 1024,
            note="Multilingual, no query instruction. The handoff's proposed default.",
        ),
        EmbeddingModel(
            "intfloat/e5-large-v2", 335, 1024,
            query_prefix="query: ", passage_prefix="passage: ",
            note="Asymmetric prefixes are mandatory for E5; omitting them is a large penalty.",
        ),
        EmbeddingModel(
            "abhinand/MedEmbed-large-v0.1", 335, 1024, query_prefix=BGE_QUERY,
            note="Medical-domain fine-tune of BGE; inherits the BGE query instruction.",
        ),
        EmbeddingModel(
            "NeuML/pubmedbert-base-embeddings", 109, 768,
            note="Biomedical PubMedBERT encoder, no instruction.",
        ),
        EmbeddingModel(
            "Alibaba-NLP/gte-base-en-v1.5", 137, 768,
            note="GTE v1.5 uses no query instruction.",
        ),
        EmbeddingModel(
            "ncbi/MedCPT", 109, 768,
            note="Asymmetric PAIR of encoders, not one model with prefixes - see "
                 "retrieval/medcpt.py. Trained on 255M PubMed click logs, so the query "
                 "side has seen how people actually type. Tested because every other "
                 "model here was compared on the question-first gold set, whose "
                 "phrasing gap is too small to distinguish them.",
        ),
    ]
}


DEFAULT_ABLATION = [
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-large-en-v1.5",
    "BAAI/bge-m3",
    "intfloat/e5-large-v2",
    "abhinand/MedEmbed-large-v0.1",
    "NeuML/pubmedbert-base-embeddings",
]


def get(name: str) -> EmbeddingModel:
    """Look up a model, defaulting to no prefixes for anything unregistered."""
    if name in REGISTRY:
        return REGISTRY[name]
    return EmbeddingModel(name, 0, 0, note="Unregistered; assuming no prefix.")


def slug(name: str) -> str:
    return name.replace("/", "__").replace(".", "_")
