import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None


def _to_dense_matrix(sparse_matrix, n_components=128):
    if sparse_matrix is None or sparse_matrix.shape[0] == 0:
        return np.empty((0, 0), dtype=np.float32), None

    if sparse_matrix.shape[0] < 3 or sparse_matrix.shape[1] < 3:
        dense = sparse_matrix.toarray().astype(np.float32)
        return normalize(dense), None

    max_components = max(2, min(n_components, sparse_matrix.shape[1] - 1)) if sparse_matrix.shape[1] > 2 else sparse_matrix.shape[1]
    if max_components <= 0:
        dense = sparse_matrix.toarray().astype(np.float32)
        return normalize(dense), None

    svd = TruncatedSVD(n_components=max_components, random_state=42)
    dense = svd.fit_transform(sparse_matrix).astype(np.float32)
    dense = normalize(dense)
    return dense, svd


class CaseKnowledgeStore:
    """
    Hybrid vector knowledge store with FAISS support and sklearn fallback.
    """

    def __init__(self, document_ids, sparse_matrix):
        self.document_ids = list(document_ids)
        self.sparse_matrix = sparse_matrix
        self.document_vectors, self.svd = _to_dense_matrix(sparse_matrix)
        self.backend = "empty"
        self.index = None

        if len(self.document_ids) == 0 or self.document_vectors.size == 0:
            return

        if faiss is not None:
            self.backend = "faiss"
            self.index = faiss.IndexFlatIP(self.document_vectors.shape[1])
            self.index.add(self.document_vectors.astype(np.float32))
        else:
            self.backend = "sklearn"
            self.index = NearestNeighbors(metric="cosine", algorithm="brute")
            self.index.fit(self.document_vectors)

    def _vectorize_query(self, query_vector):
        if self.svd is not None:
            dense_query = self.svd.transform(query_vector).astype(np.float32)
        else:
            dense_query = query_vector.toarray().astype(np.float32)
        return normalize(dense_query)

    def search(self, query_vector, top_k=5):
        if self.index is None or len(self.document_ids) == 0:
            return pd.DataFrame(columns=["retriever_document_id", "vector_score"])

        dense_query = self._vectorize_query(query_vector)

        if self.backend == "faiss":
            scores, indices = self.index.search(dense_query.astype(np.float32), top_k)
            rows = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue
                rows.append(
                    {
                        "retriever_document_id": self.document_ids[idx],
                        "vector_score": float(score),
                    }
                )
            return pd.DataFrame(rows)

        distances, indices = self.index.kneighbors(dense_query, n_neighbors=min(top_k, len(self.document_ids)))
        rows = []
        for distance, idx in zip(distances[0], indices[0]):
            rows.append(
                {
                    "retriever_document_id": self.document_ids[idx],
                    "vector_score": float(1 - distance),
                }
            )
        return pd.DataFrame(rows)

    def summary(self):
        return {
            "knowledge_store_backend": self.backend,
            "knowledge_store_documents": len(self.document_ids),
            "knowledge_store_vector_dim": int(self.document_vectors.shape[1]) if self.document_vectors.ndim == 2 and self.document_vectors.size else 0,
        }
