from __future__ import annotations

from datetime import datetime

from GeneticAlgorithm.Gene import Gene

from Instructor import Instructor

class Course(Gene):
    def __init__(self) -> None:
        super().__init__()
        self.name: str = None
        self.expected_enrollment: int = None
        self.preferred_instructors: dict[str, Instructor] = {}
        self.other_instructors: dict[str, Instructor] = {}

    def __eq__(self, __o: Course) -> bool:
        return super().__eq__(__o) and self.name == __o.name