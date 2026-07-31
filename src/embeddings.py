"""Embedding generation for AFR retrieval.

The encoder is synchronous and CPU-bound, so query encoding is pushed off the
event loop — otherwise one embedding call stalls the two other concurrent
requests (NFR-3.4).

The model is constructed lazily behind :func:`warm`, not at import. Importing
this module must stay free of side effects so the test suite and ``/health``
never depend on a model download; ``src/api.py`` calls :func:`warm` during
startup instead, which still pays the cost before the service reports healthy
(NFR-1.4).

Documents and queries are encoded through different entry points on purpose.
The BGE family is trained with an instruction prefix on the query side only,
and fastembed applies it in ``query_embed`` — using ``embed`` for both would
quietly cost recall.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from functools import lru_cache

import numpy as np

from src.config import load_data_paths

logger = logging.getLogger(__name__)

#: Batch size for document encoding. Large enough to keep the ONNX session
#: busy, small enough that one batch's activations stay modest on the Atom.
DOCUMENT_BATCH_SIZE = 256


@lru_cache(maxsize=1)
def _encoder():
    from fastembed import TextEmbedding

    paths = load_data_paths()
    logger.info("Loading embedding model %s", paths.embedding_model_name)
    return TextEmbedding(
        model_name=paths.embedding_model_name,
        cache_dir=str(paths.embedding_cache_dir),
    )


def warm() -> None:
    """Construct the encoder ahead of first use."""
    _encoder()


def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    """Scale each row to unit length so cosine similarity is a plain dot product.

    Zero-length rows are left alone rather than producing NaN: an article whose
    encoded text was empty should score zero against every query, not poison
    the whole similarity matrix.
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    np.divide(matrix, norms, out=matrix, where=norms > 0)
    return matrix


def embed_documents(
    texts: Sequence[str], *, batch_size: int = DOCUMENT_BATCH_SIZE
) -> np.ndarray:
    """Encode article text for indexing.

    Synchronous and blocking by design — the only caller is ``src/ingest.py``,
    which runs offline.

    Returns:
        ``float32[len(texts), dim]``, L2-normalised.
    """
    vectors = list(_encoder().embed(list(texts), batch_size=batch_size))
    if not vectors:
        return np.zeros((0, 0), dtype=np.float32)
    return _l2_normalise(np.asarray(vectors, dtype=np.float32))


def embed_query(text: str) -> np.ndarray:
    """Encode one query string. Synchronous; prefer :func:`aembed_query`.

    Returns:
        ``float32[dim]``, L2-normalised.
    """
    vector = np.asarray(next(iter(_encoder().query_embed([text]))), dtype=np.float32)
    return _l2_normalise(vector.reshape(1, -1))[0]


async def aembed_query(text: str) -> np.ndarray:
    """Encode one query string without blocking the event loop (NFR-3.4)."""
    return await asyncio.to_thread(embed_query, text)


def embedding_dim() -> int:
    """Dimensionality of the configured model, determined by encoding a probe.

    Read from the model rather than hard-coded, so switching
    ``EMBEDDING_MODEL_NAME`` cannot silently produce a vector file whose shape
    disagrees with the query encoder.
    """
    return int(embed_query("dimension probe").shape[0])
