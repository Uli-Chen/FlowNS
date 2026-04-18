"""FAISS-based approximate nearest neighbor index for item embeddings."""

import logging
import threading

import numpy as np

logger = logging.getLogger(__name__)

try:
    import faiss

    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False
    logger.warning(
        "faiss is not installed. Falling back to NumPy brute-force search. "
        "Install faiss-cpu or faiss-gpu for much faster retrieval."
    )


class _NumpyFallbackIndex:
    """CPU fallback using NumPy dot product when FAISS is unavailable."""

    def __init__(self, d: int, metric: str):
        self.d = d
        self.metric = metric
        self._embeddings: np.ndarray | None = None
        self._lock = threading.Lock()

    @property
    def ntotal(self) -> int:
        return 0 if self._embeddings is None else self._embeddings.shape[0]

    def reset(self) -> None:
        self._embeddings = None

    def add(self, embeddings: np.ndarray) -> None:
        self._embeddings = embeddings.copy()

    def search(self, queries: np.ndarray, k: int):
        with self._lock:
            if self._embeddings is None or self._embeddings.shape[0] == 0:
                batch = queries.shape[0]
                return np.full((batch, k), -1.0, dtype=np.float32), np.full(
                    (batch, k), -1, dtype=np.int64
                )

            if self.metric == "IP":
                # Inner product: higher is better.
                scores = queries @ self._embeddings.T  # [batch, n_items]
                k_clamped = min(k, scores.shape[1])
                indices = np.argpartition(-scores, k_clamped - 1, axis=1)[
                    :, :k_clamped
                ]
                # Sort the top-k by descending score.
                row_scores = np.take_along_axis(scores, indices, axis=1)
                order = np.argsort(-row_scores, axis=1)
                indices = np.take_along_axis(indices, order, axis=1)
                distances = np.take_along_axis(row_scores, order, axis=1)
            else:
                # L2: lower is better.
                # ||q - e||^2 = ||q||^2 + ||e||^2 - 2 q . e
                q_sq = np.sum(queries ** 2, axis=1, keepdims=True)  # [batch, 1]
                e_sq = np.sum(self._embeddings ** 2, axis=1)  # [n_items]
                dists = q_sq + e_sq - 2.0 * (queries @ self._embeddings.T)
                np.maximum(dists, 0.0, out=dists)  # numerical safety

                k_clamped = min(k, dists.shape[1])
                indices = np.argpartition(dists, k_clamped - 1, axis=1)[
                    :, :k_clamped
                ]
                row_dists = np.take_along_axis(dists, indices, axis=1)
                order = np.argsort(row_dists, axis=1)
                indices = np.take_along_axis(indices, order, axis=1)
                distances = np.take_along_axis(row_dists, order, axis=1)

            # Pad if k > n_items.
            if k_clamped < k:
                batch = queries.shape[0]
                pad_width = k - k_clamped
                indices = np.pad(
                    indices, ((0, 0), (0, pad_width)), constant_values=-1
                )
                fill = -1.0 if self.metric == "IP" else float("inf")
                distances = np.pad(
                    distances, ((0, 0), (0, pad_width)), constant_values=fill
                )

            return distances.astype(np.float32), indices.astype(np.int64)


class FAISSIndex:
    """Approximate nearest neighbor index backed by FAISS.

    Parameters
    ----------
    d : int
        Embedding dimensionality.
    metric : str
        ``'IP'`` for inner product (cosine after L2-normalisation) or
        ``'L2'`` for Euclidean distance.
    use_gpu : bool
        If *True* and a GPU-capable FAISS build is available, the index is
        placed on GPU 0.
    """

    def __init__(self, d: int, metric: str = "IP", use_gpu: bool = False):
        if metric not in ("IP", "L2"):
            raise ValueError(f"metric must be 'IP' or 'L2', got '{metric}'")

        self.d = d
        self.metric = metric
        self.use_gpu = use_gpu
        self._lock = threading.Lock()

        if _FAISS_AVAILABLE:
            self._index = self._make_index()
        else:
            self._index = _NumpyFallbackIndex(d, metric)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_index(self):
        """Create a fresh FAISS index (CPU or GPU)."""
        if self.metric == "IP":
            index = faiss.IndexFlatIP(self.d)
        else:
            index = faiss.IndexFlatL2(self.d)

        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
                # Keep a reference so the resources are not garbage-collected.
                self._gpu_resources = res
            except Exception:
                logger.warning(
                    "GPU requested but faiss GPU conversion failed. "
                    "Falling back to CPU."
                )

        return index

    @staticmethod
    def _as_float32_contiguous(arr: np.ndarray) -> np.ndarray:
        """Ensure *arr* is C-contiguous float32 (required by FAISS)."""
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        return arr

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, item_embeddings: np.ndarray) -> None:
        """Reset and populate the index with *item_embeddings*.

        Parameters
        ----------
        item_embeddings : np.ndarray
            Array of shape ``[n_items, d]`` with dtype convertible to
            ``np.float32``.
        """
        item_embeddings = self._as_float32_contiguous(item_embeddings)

        if item_embeddings.ndim != 2 or item_embeddings.shape[1] != self.d:
            raise ValueError(
                f"Expected embeddings of shape [n_items, {self.d}], "
                f"got {item_embeddings.shape}"
            )

        if self.metric == "IP" and _FAISS_AVAILABLE:
            faiss.normalize_L2(item_embeddings)
        elif self.metric == "IP":
            norms = np.linalg.norm(item_embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            item_embeddings = item_embeddings / norms

        with self._lock:
            if _FAISS_AVAILABLE:
                self._index = self._make_index()
            else:
                self._index.reset()
            self._index.add(item_embeddings)

    def search(
        self, queries: np.ndarray, M: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Find the *M* nearest neighbours for each query.

        Parameters
        ----------
        queries : np.ndarray
            Query vectors of shape ``[batch, d]`` (dtype convertible to
            ``np.float32``).
        M : int
            Number of neighbours to retrieve per query.

        Returns
        -------
        distances : np.ndarray
            Shape ``[batch, M]``.  Inner-product similarities (descending) or
            L2 distances (ascending).
        indices : np.ndarray
            Shape ``[batch, M]``.  Item indices into the array that was passed
            to :meth:`build`.
        """
        queries = self._as_float32_contiguous(queries)

        if queries.ndim != 2 or queries.shape[1] != self.d:
            raise ValueError(
                f"Expected queries of shape [batch, {self.d}], "
                f"got {queries.shape}"
            )

        if self.metric == "IP" and _FAISS_AVAILABLE:
            faiss.normalize_L2(queries)
        elif self.metric == "IP":
            norms = np.linalg.norm(queries, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            queries = queries / norms

        with self._lock:
            distances, indices = self._index.search(queries, M)

        return distances, indices

    @property
    def ntotal(self) -> int:
        """Number of vectors currently stored in the index."""
        return self._index.ntotal
