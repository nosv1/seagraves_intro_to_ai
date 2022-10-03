# Chris Seagraves
# Course Scheduling with a Genetic Algorithm

# python imports
from dataclasses import dataclass, fields
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from random import choice, random

# GeneticAlgorithm imports
from GeneticAlgorithm.Chromosome import Chromosome
from GeneticAlgorithm.Gene import Gene
from GeneticAlgorithm.GeneticAlgorithm import GeneticAlgorithm

# support imports
from ClassTime import ClassTime
from Course import Course
from Instructor import Instructor
from Room import Room

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Convert DataFrames to lists of objects

def create_courses(courses_dataframe: pd.DataFrame) -> list[Course]: 
    courses: list[Course] = []
    for i, row in courses_dataframe.iterrows():
        course: Course = Course()
        course.name = row["name"]
        course.expected_enrollment = int(row["expected_enrollment"])
        for j in range(1, 10 + 1):  # + 10 because we allot 10 preferred and other instructors per course
            if row[f"preferred_instructor_{j}"] != "":
                instructor: Instructor = Instructor(name=row[f"preferred_instructor_{j}"])
                course.preferred_instructors[instructor.name] = instructor
            if row[f"other_instructor_{j}"] != "":
                instructor: Instructor = Instructor(name=row[f"other_instructor_{j}"])
                course.other_instructors[instructor.name] = instructor
        courses.append(course)
    return courses
    
def create_class_times(class_times_dataframe: pd.DataFrame) -> list[ClassTime]:
    class_times: list[ClassTime] = []
    for i, row in class_times_dataframe.iterrows():
        class_times.append(
            ClassTime(
                start=datetime.strptime(row["start"], "%H:%M"),
                end=datetime.strptime(row["end"], "%H:%M")
            )
        )
    return class_times

def create_rooms(rooms_dataframe: pd.DataFrame) -> list[Room]:
    rooms: list[Room] = []
    for i, row in rooms_dataframe.iterrows():
        room = Room(building=row["building"], room=row["room"], capacity=int(row["capacity"]))
        rooms.append(room)
    return rooms

def create_instructors(instructors_dataframe: pd.DataFrame) -> list[Instructor]:
    instructors: list[Instructor] = []
    for i, row in instructors_dataframe.iterrows():
        instructor: Instructor = Instructor(name=row["name"])
        instructors.append(instructor)
    return instructors

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Generate Genes

def generate_course_gene(course: Course) -> Gene:
    return course

def generate_room_gene(rooms: list[Room]) -> Gene:
    return choice(rooms)
    
def generate_class_time_gene(class_times: list[ClassTime]) -> Gene:
    return choice(class_times)

def generate_instructor_gene(instructors: list[Instructor]) -> Gene:
    return choice(instructors)

def generate_chromosomes(
    possible_genes: dict[type, list[Gene]],
    count: int=1
) -> list[Chromosome]:
    chromosomes: list[Chromosome] = []
    for i in range(count):
        chromosome: Chromosome = Chromosome(genes=[])
        for course in possible_genes[Course]:
            chromosome.genes.append(generate_course_gene(course))
            chromosome.genes.append(generate_class_time_gene(possible_genes[ClassTime]))
            chromosome.genes.append(generate_instructor_gene(possible_genes[Instructor]))
            chromosome.genes.append(generate_room_gene(possible_genes[Room]))
        chromosomes.append(chromosome)
    return chromosomes

def mutate_chromosome(
    chromosome: Chromosome,
    mutation_rate: float, 
    possible_genes: dict[type, list[Gene]]
) -> Chromosome:
    """
    Creates a new chromosome by randomly selecting genes from the parents.
    """
    mutation: Chromosome = Chromosome(genes=[])
    for i, gene in enumerate(chromosome.genes):
        if random() < mutation_rate:
            if isinstance(gene, Course):
                # Course's encode the current sub-chromosome;
                # We don't change them.
                pass
            elif isinstance(gene, ClassTime):
                gene = generate_class_time_gene(possible_genes[ClassTime])
            elif isinstance(gene, Instructor):
                gene = generate_instructor_gene(possible_genes[Instructor])
            elif isinstance(gene, Room):
                gene = generate_room_gene(possible_genes[Room])
        mutation.genes.append(gene)
    return mutation

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Evaluate Chromosome

def evaluate_chromosome(chromosome: Chromosome, print_checks=False) -> float:

    """
    The idea with these checks is to loop through the genes, storing the 
    values we care about until we get to the next course, then evaluating the 
    stored values.
    We KNOW Course goes first, but the order of the following genes isn't a given.
    """

    # It should be noted, this dataclass is not needed, but for the sake of 
    # displaying the data of what checks succeeded and failed, it's useful.
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
        instructor_two_classes_one_time: float = 0.0
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

    checks: Checks = Checks()

    def same_room_same_time_check() -> None:
        """
        Class is scheduled at the same time in the same room as another class: -0.5
        """
        rooms: dict[str, set[datetime]] = {}
        room: Room = None
        class_time: ClassTime = None
        for gene in chromosome.genes:
            if isinstance(gene, Room):
                room = gene
                rooms[room.key] = set()

            elif isinstance(gene, ClassTime):
                class_time = gene
                class_time = class_time.start

            elif isinstance(gene, Course):                
                if room and class_time:
                    if class_time not in rooms[room.key]:
                        rooms[room.key].add(class_time)

                    else:
                        checks.same_room_same_time -= 0.5
    
    def room_size_check() -> None:
        """
        Room size:
        - Class is in a room too small for its expected enrollment: -0.5
        - Class is in a room with capacity > 3 times expected enrollment: -0.2
        - Class is in a room with capacity > 6 times expected enrollment: -0.4
        - Otherwise +0.3
        """
        course: Course = None
        room: Room = None
        for gene in chromosome.genes:
            if isinstance(gene, Course):
                course = gene

            elif isinstance(gene, Room):
                room = gene

            if room and course:

                if room.capacity < course.expected_enrollment:
                    checks.room_too_small -= 0.5

                elif room.capacity >= course.expected_enrollment * 3:
                    checks.room_3x_too_big -= 0.2

                elif room.capacity >= course.expected_enrollment * 6:
                    checks.room_6x_too_big -= 0.4

                else:
                    checks.room_size_sufficient += 0.3

                # set to None so we don't the loop again for the same course
                course = None
                room = None

    def preferred_instructor_check() -> None:
        """
        Class is taught by a preferred faculty member: +0.5
        Class is taught by another faculty member listed for that course: +0.2
        Class is taught by some other faculty: -0.1
        """
        course: Course = None
        instructor: Instructor = None
        for gene in chromosome.genes:
            if isinstance(gene, Course):
                course = gene

            elif isinstance(gene, Instructor):
                instructor = gene

            if course and instructor:

                if instructor.name in course.preferred_instructors:
                    checks.preferred_instructor += 0.5

                elif instructor.name in course.other_instructors:
                    checks.other_instructor += 0.2

                else:
                    checks.other_faculty -= 0.1

                # set to None so we don't the loop again for the same course
                course = None
                instructor = None
        
    def instructor_load_check() -> None:
        """
        Instructor load:
        - Course instructor is scheduled for only 1 class in this time slot: +0.2
        - Course instructor is scheduled for more than one class at the same time: -0.2
        - Instructor is scheduled to teach more than 4 classes total: -0.5
        - Instructor is scheduled to teach 1 or 2 classes: -0.4
            -  no penalty for Xu for teaching < 3 courses
        - Consecutive time slots: +0.5
        - If consective time slots and one class is in Bloch or Katz and the 
            other isn't: -0.4
        """
        instructor: Instructor = None
        class_time: ClassTime = None
        room: Room = None
        course: Course = None
        course_count: int = 0
        for gene in chromosome.genes:
            if isinstance(gene, Instructor):
                instructor = gene
                course_count = 1

            elif isinstance(gene, ClassTime):
                class_time = gene
                
            elif isinstance(gene, Room):
                room = gene

            elif isinstance(gene, Course):
                course = gene

            if instructor and class_time and room and course:
                check_instructor: Instructor = None
                check_class_time: ClassTime = None
                check_room: Room = None
                check_course: Course = None
                one_class_one_time = True
                consecutive_time_slots = False
                far_away_rooms = False
                for check_gene in chromosome.genes:
                    if check_course and course == check_course:
                        continue

                    elif isinstance(check_gene, Instructor):
                        check_instructor = check_gene

                        # This doesn't double count, because we skip loop if 
                        # we're in the same course.
                        if instructor == check_instructor:
                            course_count += 1 

                    elif isinstance(check_gene, ClassTime):
                        check_class_time = check_gene

                    elif isinstance(check_gene, Room):
                        check_room = check_gene

                    elif isinstance(check_gene, Course):
                        check_course = check_gene

                    if check_class_time and check_instructor and check_room and check_course:
                        if (
                            check_instructor == instructor and 
                            check_class_time == class_time
                        ):
                            one_class_one_time = False
                        
                        if check_class_time.start - class_time.start == timedelta(hours=1):
                            consecutive_time_slots = True

                            if (
                                room.building in {"Bloch", "Katz"} and
                                check_room.building not in {"Bloch", "Katz"}
                            ):
                                far_away_rooms = True

                ## course count
                if course_count > 4:
                    checks.instructor_more_than_4_classes -= 0.5

                elif course_count < 3:
                    if instructor.name != "Xu":
                        checks.instructor_less_than_3_classes -= 0.4

                ## one class one time
                if one_class_one_time:
                    checks.instructor_one_class_one_time += 0.2
                ## multiple class same time
                else:
                    # I THINK by deduction, this makes sense
                    checks.instructor_two_classes_one_time -= 0.2

                ## consecutive time slots
                if consecutive_time_slots:
                    if far_away_rooms:
                        checks.instructor_consecutive_slots_far_away_rooms -= 0.4
                    else:
                        checks.instructor_consecutive_slots += 0.5

                # set to None so we don't the loop again for the same course
                instructor = None
                class_time = None
                room = None
                course = None

    def course_specific_check() -> None:
        """
        - The 2 sections of CS 101 are more than 4 hours apart: +0.5
        - Both sections of CS 101 are in the same time slot: -0.5
        - The 2 sections of CS 191 are more than 4 hours apart: +0.5
        - Both sections of CS 191 are in the same time slot: -0.5
        - A section of CS 191 and a section of CS 101 are taught in 
            consecutive time slots: +0.5
        - If consective time slots and one class is in Bloch or Katz and the 
            other isn't: -0.4
        - A section of CS 191 and a section of CS 101 are taught separated by
            1 hour: +0.25
        - A section of CS 191 and a section of CS 101 are taught in the same
            time slot: -0.25
        """
        course: Course = None
        class_time: ClassTime = None
        room: Room = None
        for gene in chromosome.genes:
            if isinstance(gene, Course):
                course = gene

            elif isinstance(gene, ClassTime):
                class_time = gene

            elif isinstance(gene, Room):
                room = gene

            if course and class_time and room:
                check_course: Course = None
                check_class_time: ClassTime = None
                check_room: Room = None
                consecutive_time_slots = False
                one_hour_gap = False
                same_time = False
                far_away_rooms = False
                four_hour_gap = False
                for check_gene in chromosome.genes:
                    if check_course and course == check_course:
                        continue

                    elif isinstance(check_gene, Course):
                        check_course = check_gene

                    elif isinstance(check_gene, ClassTime):
                        check_class_time = check_gene

                    elif isinstance(check_gene, Room):
                        check_room = check_gene

                    if check_course and check_class_time and room:

                        if check_class_time.start - class_time.end == timedelta(hours=0):
                            same_time = True
                        elif check_class_time.start - class_time.end == timedelta(hours=1):
                            consecutive_time_slots = True

                            if (
                                room.building in {"Bloch", "Katz"} and
                                check_room.building not in {"Bloch", "Katz"}
                            ):
                                far_away_rooms = True

                        elif check_class_time.start - class_time.end == timedelta(hours=2):
                            one_hour_gap = True

                        elif check_class_time.start - class_time.end == timedelta(hours=4):
                            four_hour_gap = True

                        # section following or followed by another section
                        if (
                            "CS101" in course.name and "CS191" in check_course.name or
                            "CS101" in check_course.name and "CS191" in course.name
                        ):
                            if consecutive_time_slots:
                                if far_away_rooms:
                                    checks.sections_consecutive_far_away_rooms -= 0.4
                                else:
                                    checks.cs_101_191_consecutive_slots += 0.5

                            elif one_hour_gap:
                                checks.cs_101_191_one_hour_apart += 0.25

                            elif same_time:
                                checks.cs_101_191_same_time -= 0.25

                        # both sections
                        elif (
                            "CS101" in course.name and "CS101" in check_course.name or
                            "CS191" in course.name and "CS191" in check_course.name
                        ):
                            if same_time:
                                # These could be condensed if we weren't storing
                                # info on what specific course.
                                if "CS101" in course.name:
                                    checks.cs_101_same_time -= 0.5
                                elif "CS191" in course.name:
                                    checks.cs_191_same_time -= 0.5

                            elif four_hour_gap:
                                if "CS101" in course.name:
                                    checks.cs_101_4_hours_apart += 0.5
                                elif "CS191" in course.name:
                                    checks.cs_191_4_hours_apart += 0.5
                                

                # set to None so we don't the loop again for the same course
                course = None
                class_time = None
                room = None     

    same_room_same_time_check()
    room_size_check()
    preferred_instructor_check()
    instructor_load_check()
    course_specific_check()

    for check in fields(checks):
        attr: float = getattr(checks, check.name)
        chromosome.fitness += attr

        if print_checks:
            print(f"    {check.name}: {attr:.2f}")

    if print_checks:
        print("-" * 40)
        print(f"Fitness: {chromosome.fitness:.2f}")

    return chromosome

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Display Chromosome

def display_chromosome(chromosome: Chromosome) -> None:
    """
    Displays a chromosome in a readable format.
    
    Course: Name
    Class Time: Start - End hh:mm am/pm
    Instructor: Name
    Room: Building (Room) expected/seats  

    \--------------------------------------------------
    Fitness: 0.00
    """
    course: Course = None
    class_time: ClassTime = None
    instructor: Instructor = None
    room: Room = None
    print("-" * 40)
    for gene in chromosome.genes:
        if isinstance(gene, Course):
            course = gene
            print(f"\n\nCourse: {course.name}")

        elif isinstance(gene, ClassTime):
            class_time = gene
            print(f"Class Time: {class_time.start.strftime('%I:%M %p')} - {class_time.end.strftime('%I:%M %p')}")
        
        elif isinstance(gene, Instructor):
            instructor = gene
            print(f"Instructor: {instructor.name}")
        
        elif isinstance(gene, Room):
            room = gene
            print(f"Room: {gene.building} ({gene.room}) {course.expected_enrollment}/{room.capacity} ({course.expected_enrollment / room.capacity * 100:.2f}% full)")
    print("-" * 40)
    print(f"Fitness: {chromosome.fitness:.2f}")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# main()

def main() -> None:
    # read the data
    courses: pd.DataFrame = pd.read_csv("Database/courses.csv", dtype=str)
    class_times: pd.DataFrame = pd.read_csv("Database/class_times.csv", dtype=str)
    rooms: pd.DataFrame = pd.read_csv("Database/rooms.csv", dtype=str)
    faculty: pd.DataFrame = pd.read_csv("Database/faculty.csv", dtype=str)

    # create the objects
    possible_genes: dict[type, list[Gene]] = {
        Course: create_courses(courses),
        ClassTime: create_class_times(class_times),
        Room: create_rooms(rooms),
        Instructor: create_instructors(faculty)
    }

    # initialize the ga
    scheduling_ga: GeneticAlgorithm = GeneticAlgorithm(
        population_size=500,
        mutation_rate=0.05,
        possible_genes=possible_genes,
        chromosome_generator=generate_chromosomes,
        chromosome_evaluator=evaluate_chromosome,
        chromosome_displayer=display_chromosome,
        chromosome_mutator=mutate_chromosome,
    )
    scheduling_ga.initialize_population(count=scheduling_ga.population_size)

    # setup plot
    fig = plt.figure()
    average_fitness_subplot = fig.add_subplot(1, 1, 1)
    standard_deviation_subplot = average_fitness_subplot.twinx()
    average_fitness_subplot.set_xlabel("Generation")
    average_fitness_subplot.set_ylabel("Average Fitness")
    standard_deviation_subplot.set_ylabel("Standard Deviation", rotation=270, labelpad=15)

    # run the ga
    for i in range(20):
        print(f"\nGeneration {i}")
        scheduling_ga.evaluate_chromosomes()

        scheduling_ga.calculate_probabilities()
        scheduling_ga.create_offspring()

        print(f"Average Fitness: {scheduling_ga.average_fitness:.2f}")
        print(f"Standard Deviation: {scheduling_ga.standard_deviation:.2f}")

        average_fitness_subplot.plot(i, scheduling_ga.average_fitness, "bo")
        standard_deviation_subplot.plot(i, scheduling_ga.standard_deviation, "ro")

    scheduling_ga.evaluate_chromosomes()

    print(f"\nGeneration {i+1}")
    print(f"Average Fitness: {scheduling_ga.average_fitness:.2f}")
    print(f"Standard Deviation: {scheduling_ga.standard_deviation:.2f}")

    fittest_chromosome: Chromosome = scheduling_ga.fittest_chromosome
    scheduling_ga.display_chromosome(fittest_chromosome)
    print("\nChecks:")
    fittest_chromosome.fitness = 0
    evaluate_chromosome(fittest_chromosome, print_checks=True)

    average_fitness_subplot.legend(
        handles=[
            mpatches.Patch(color="blue", label="Average Fitness"),
            mpatches.Patch(color="red", label="Standard Deviation")
        ]
    )

    plt.show()

if __name__ == "__main__":
    main()