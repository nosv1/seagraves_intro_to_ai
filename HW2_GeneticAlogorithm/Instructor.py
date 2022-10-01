from __future__ import annotations

from GeneticAlgorithm.Gene import Gene

class Instructor(Gene):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def __eq__(self, __o: Instructor) -> bool:
        return super().__eq__(__o) and self.name == __o.name