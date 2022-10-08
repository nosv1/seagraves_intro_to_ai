from dataclasses import dataclass

from GeneticAlgorithm.Chromosome import Chromosome

class Check:
    def __init__(self, count: int = 0, weight: float = 0.0) -> None:
        self.count = count
        self.weight = weight

        self.score = 0

class Checks:
    def __init__(self) -> None:
        self.same_room_same_time: Check = Check(weight=-0.5)

        self.room_too_small: Check = Check(weight=-0.5)
        self.room_3x_too_big: Check = Check(weight=-0.2)
        self.room_6x_too_big: Check = Check(weight=-0.4)
        self.room_size_sufficient: Check = Check(weight=0.3)

        self.preferred_instructor: Check = Check(weight=0.5)
        self.other_instructor: Check = Check(weight=0.2)
        self.other_faculty: Check = Check(weight=-0.1)

        self.instructor_one_class_one_time: Check = Check(weight=0.2)
        self.instructor_multiple_classes_one_time: Check = Check(weight=-0.2)
        self.instructor_more_than_4_classes: Check = Check(weight=-0.5)
        self.instructor_less_than_3_classes: Check = Check(weight=-0.4)
        self.instructor_consecutive_slots: Check = Check(weight=0.5)
        self.instructor_consecutive_slots_far_away_rooms: Check = Check(weight=-0.4)

        self.cs_101_4_hour_gap: Check = Check(weight=0.5)
        self.cs_101_same_time: Check = Check(weight=-0.5)
        self.cs_191_4_hour_gap: Check = Check(weight=0.5)
        self.cs_191_same_time: Check = Check(weight=-0.5)
        self.cs_101_191_consecutive: Check = Check(weight=0.5)
        self.sections_consecutive_far_away_rooms: Check = Check(weight=-0.4)
        self.cs_101_191_one_hour_gap: Check = Check(weight=0.25)
        self.cs_101_191_same_time: Check = Check(weight=-0.25)

    def print_checks(self):
        for check_name, check in self.__dict__.items():
            check: Check = check
            print(f"{check_name}: {check.count} * {check.weight} = {check.score}")

class Chromosome(Chromosome):
    # It should be noted, this dataclass is not needed, but for the sake of 
    # displaying the data of what checks succeeded and failed, it's useful.
    # Without this, we could juste update chromosome.fitness when we do the checks.

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.checks = Checks()

    def set_fitness_from_checks(self) -> None:
        self.fitness = 0
        for check_name in self.checks.__dict__:
            check: Check = getattr(self.checks, check_name)
            check.score = check.count * check.weight
            self.fitness += check.score