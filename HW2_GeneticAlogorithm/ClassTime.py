from __future__ import annotations

from GeneticAlgorithm.Gene import Gene

from datetime import datetime

class ClassTime(Gene):
    def __init__(self, start: datetime, end: datetime) -> None:
        super().__init__()
        self.start = start
        self.end = end

    def __eq__(self, __o: ClassTime) -> bool:
        return (
            super().__eq__(__o) and 
            self.start == __o.start and 
            self.end == __o.end
        )