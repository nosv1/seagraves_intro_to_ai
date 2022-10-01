from __future__ import annotations

class Gene:
    def __init__(self) -> None:
        self.fitness: float = 0

    def __eq__(self, __o: Gene) -> bool:
        return True