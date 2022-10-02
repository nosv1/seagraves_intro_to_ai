from __future__ import annotations

from .Gene import Gene

class Chromosome:
    def __init__(self, genes: list[Gene]) -> None:
        self.genes = genes

        self.fitness: float = 0.0