from dataclasses import dataclass

from GeneticAlgorithm.Chromosome import Chromosome


class Chromosome(Chromosome):
    # It should be noted, this dataclass is not needed, but for the sake of 
    # displaying the data of what checks succeeded and failed, it's useful.
    # Without this, we could juste update chromosome.fitness when we do the checks.
    @dataclass
    class Checks:
        same_room_same_time: float = 0.0

        room_too_small: float = 0.0
        room_3x_too_big: float = 0.0
        room_6x_too_big: float = 0.0
        room_size_sufficient: float = 0.0

        preferred_instructor: float = 0.0
        other_instructor: float = 0.0
        other_faculty: float = 0.0

        instructor_one_class_one_time: float = 0.0
        instructor_multiple_classes_one_time: float = 0.0
        instructor_more_than_4_classes: float = 0.0
        instructor_less_than_3_classes: float = 0.0
        instructor_consecutive_slots: float = 0.0
        instructor_consecutive_slots_far_away_rooms: float = 0.0

        cs_101_4_hours_apart: float = 0.0
        cs_101_same_time: float = 0.0
        cs_191_4_hours_apart: float = 0.0
        cs_191_same_time: float = 0.0
        cs_101_191_consecutive: float = 0.0
        sections_consecutive_far_away_rooms: float = 0.0
        cs_101_191_one_hour_apart: float = 0.0
        cs_101_191_same_time: float = 0.0

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.checks = self.Checks()

    def set_fitness_from_checks(self) -> float:
        self.fitness = sum(
            self.checks.__getattribute__(key)
            for key
            in self.checks.__dict__.keys()
        )