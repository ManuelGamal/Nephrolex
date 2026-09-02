"""MedCPT encoders, wired to look like a SentenceTransformer to the rest of the code.

Why this file exists rather than a registry entry
-------------------------------------------------
Every other model in `embedding_models.py` is one set of weights that encodes both
passages and queries, differing only by an instruction prefix. MedCPT is not: it is a
*pair* of separately-trained encoders, `ncbi/MedCPT-Query-Encoder` and
`ncbi/MedCPT-Article-Encoder`, trained together on 255 million PubMed click logs so
that a real user's query lands near the formal article they clicked. That asymmetry is
the entire reason it is worth testing here - the query side has seen how people
actually type, which is the gap this project measured between echo-phrased questions
(nDCG 0.90) and clinician-phrased ones (0.38).

Two traps, both of which would have measured the harness instead of the model:

  pooling   Neither repository ships a `modules.json` or `1_Pooling/` directory, so
            `SentenceTransformer` would fall back to mean pooling. MedCPT is trained
            with [CLS] pooling. Mean-pooling it produces a working, plausible, wrong
            number - the same failure mode as evaluating a prefixed model without its
            prefix, which `embedding_models.py` already warns about.

  lengths   The query encoder was trained at 64 tokens and the article encoder at 512.
            Encoding queries at 512 is harmless but slow; encoding articles at 64
            silently truncates most of this corpus.

Article input format
--------------------
MedCPT's article encoder was trained on [title, abstract] pairs, tokenized as a
sequence pair rather than one concatenated string. The corpus already stores its text
as "KDIGO | SECTION HEADING\\nbody...", so the first line is used as the title and the
remainder as the abstract. That is a closer match to the training distribution than
flattening it, and it costs nothing.

Similarity
----------
MedCPT scores with a raw dot product of unnormalised [CLS] vectors. This project's
fusion min-max normalises every signal over the candidate pool, and the dense signal is
combined with others that are cosine-scaled, so vectors are L2-normalised here for
comparability. That makes the dense score a cosine rather than MedCPT's native dot
product - a deliberate deviation, recorded because it could account for a difference
and because the alternative is mixing incommensurable scales inside the fusion.
"""

from __future__ import annotations

import numpy as np

QUERY_MODEL = "ncbi/MedCPT-Query-Encoder"
ARTICLE_MODEL = "ncbi/MedCPT-Article-Encoder"

# The logical name recorded in the embeddings sidecar and passed as --dense-model.
# A single name for the pair, so the guard in retrieve() that refuses mismatched
# vectors still has one thing to compare.
LOGICAL_NAME = "ncbi/MedCPT"

QUERY_MAX_TOKENS = 64
ARTICLE_MAX_TOKENS = 512


class MedCPTEncoder:
    """Exposes `.encode(...)` with the SentenceTransformer signature this codebase uses."""

    def __init__(self, role: str, device: str | None = None) -> None:
        if role not in {"query", "article"}:
            raise ValueError(f"role must be 'query' or 'article', not {role!r}")
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.role = role
        self.max_length = QUERY_MAX_TOKENS if role == "query" else ARTICLE_MAX_TOKENS
        name = QUERY_MODEL if role == "query" else ARTICLE_MODEL
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(self.device).eval()
        self._torch = torch

    def _tokenize(self, batch: list[str]):
        if self.role == "query":
            return self.tokenizer(batch, truncation=True, padding=True,
                                  max_length=self.max_length, return_tensors="pt")
        # Article encoder: feed [title, body] as a sequence pair, matching how it was
        # trained on [title, abstract].
        titles, bodies = [], []
        for text in batch:
            head, _, rest = text.partition("\n")
            if rest.strip():
                titles.append(head)
                bodies.append(rest)
            else:
                titles.append("")
                bodies.append(text)
        return self.tokenizer(titles, bodies, truncation=True, padding=True,
                              max_length=self.max_length, return_tensors="pt")

    def encode(self, texts, normalize_embeddings: bool = True, batch_size: int = 16,
               show_progress_bar: bool = False, **_ignored) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        out: list[np.ndarray] = []
        with self._torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = texts[start:start + batch_size]
                encoded = self._tokenize(batch).to(self.device)
                # [CLS] pooling, as MedCPT is trained. Not mean pooling.
                vectors = self.model(**encoded).last_hidden_state[:, 0, :]
                out.append(vectors.cpu().numpy())
                if show_progress_bar and start % (batch_size * 20) == 0:
                    print(f"    {min(start + batch_size, len(texts))}/{len(texts)}",
                          flush=True)
        matrix = np.vstack(out).astype("float32")
        if normalize_embeddings:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            matrix = matrix / np.clip(norms, 1e-12, None)
        return matrix


def is_medcpt(name: str | None) -> bool:
    return bool(name) and name.startswith("ncbi/MedCPT")
