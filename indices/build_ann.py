"""Small stand-in for the ANN builder mentioned in the design doc."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class FaissIndex:
    """Minimal container mimicking the interface of a Faiss index."""

    vectors: list[list[float]]
    m: int
    efc: int

    def search(self, queries: Iterable[list[float]], k: int = 5) -> list[list[int]]:
        """Return trivial neighbour IDs for the provided queries."""

        vectors = self.vectors
        indices: list[list[int]] = []
        for _ in queries:
            indices.append(list(range(min(k, len(vectors)))))
        return indices


def build_hnsw(vectors: "Iterable[Iterable[float]]", m: int = 32, efc: int = 400) -> FaissIndex:
    """Build a dummy HNSW index compatible with the tests."""

    materialised = [list(vector) for vector in vectors]
    return FaissIndex(vectors=materialised, m=m, efc=efc)
