# Chris Seagraves
# Course Scheduling with a Genetic Algorithm

# python imports
from copy import deepcopy
from dataclasses import dataclass, fields
from datetime import datetime, timedelta
from io import TextIOWrapper
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import random as rnd
from random import choice, random
import sys


# GeneticAlgorithm imports
# from GeneticAlgorithm.Chromosome import Chromosome
from GeneticAlgorithm.Gene import Gene
from GeneticAlgorithm.GeneticAlgorithm import GeneticAlgorithm

# support imports
from ClassTime import ClassTime
from Chromosome import Chromosome
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
    return deepcopy(course)

def generate_room_gene(rooms: list[Room]) -> Gene:
    return deepcopy(choice(rooms))
    
def generate_class_time_gene(class_times: list[ClassTime]) -> Gene:
    return deepcopy(choice(class_times))

def generate_instructor_gene(instructors: list[Instructor]) -> Gene:
    return deepcopy(choice(instructors))

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
                # we don't change them.
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

def evaluate_chromosome(
    chromosome: Chromosome, print_checks: bool = False) -> Chromosome:
    # The idea with these checks is to loop through the genes, storing the 
    # values we care about until we get to the next course, then evaluating the 
    # stored values.
    # We KNOW Course goes first, but the order of the following genes isn't a given.

    remote_buildings: set[str] = {"Bloch", "Katz"}

    def same_room_same_time_check() -> None:
        """
        Class is scheduled at the same time in the same room as another class: -0.5
        """
        rooms: dict[str, set[datetime]] = {}
        room: Room = None
        class_time: ClassTime = None
        course: Course = None
        for gene in chromosome.genes:
            if isinstance(gene, Room):
                room = gene
                if room.key not in rooms:
                    rooms[room.key] = set()

            elif isinstance(gene, ClassTime):
                class_time = gene

            elif isinstance(gene, Course):
                course = gene
                room = None
                class_time = None

            if course and room and class_time:
                if class_time.start not in rooms[room.key]:
                    rooms[room.key].add(class_time.start)

                else:
                    chromosome.checks.same_room_same_time -= 0.5
                    # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
                    # checks.same_room_same_time -= 10.0

                course = None
    
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
                room = None

            elif isinstance(gene, Room):
                room = gene

            if room and course:

                if room.capacity < course.expected_enrollment:
                    chromosome.checks.room_too_small -= 0.5

                elif room.capacity >= course.expected_enrollment * 3:
                    chromosome.checks.room_3x_too_big -= 0.2

                elif room.capacity >= course.expected_enrollment * 6:
                    chromosome.checks.room_6x_too_big -= 0.4

                else:
                    chromosome.checks.room_size_sufficient += 0.3

                # Skips til next course in chromosome
                course = None

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
                instructor = None

            elif isinstance(gene, Instructor):
                instructor = gene

            if course and instructor:

                if instructor.name in course.preferred_instructors:
                    chromosome.checks.preferred_instructor += 0.5

                elif instructor.name in course.other_instructors:
                    chromosome.checks.other_instructor += 0.2

                else:
                    chromosome.checks.other_faculty -= 0.1

                course = None
        
    def instructor_load_check() -> None:
        """
        Instructor load:
        - Course instructor is scheduled for only 1 class in this time slot: +0.2
        - Course instructor is scheduled for more than one class at the same time: -0.2
        - Instructor is scheduled to teach more than 4 classes total: -0.5
        - Instructor is scheduled to teach 1 or 2 classes: -0.4
            -  no penalty for Xu for teaching < 3 courses
        - Consecutive time slots: +0.5
        - If consective time slots and one class is in remote building Bloch or 
        Katz and the following class in the other remote building: -0.4
        """
        instructor: Instructor = None
        class_time: ClassTime = None
        room: Room = None
        course: Course = None
        course_count: int = 0
        instructor_counts: dict[str, int] = {}
        for i, gene in enumerate(chromosome.genes):
            if isinstance(gene, Instructor):
                instructor = gene
                if instructor.name not in instructor_counts:
                    instructor_counts[instructor.name] = 1
                else:
                    instructor_counts[instructor.name] += 1

            elif isinstance(gene, ClassTime):
                class_time = gene
                
            elif isinstance(gene, Room):
                room = gene

            elif isinstance(gene, Course):
                course = gene
                instructor = None
                class_time = None
                room = None

            if instructor and class_time and room and course:
                check_instructor: Instructor = None
                check_class_time: ClassTime = None
                check_room: Room = None
                check_course: Course = None

                one_class_one_time = True
                consecutive_time_slots = False
                far_away_rooms = False
                for check_gene in chromosome.genes[i+1:]:
                    if isinstance(check_gene, Instructor):
                        check_instructor = check_gene

                    elif isinstance(check_gene, ClassTime):
                        check_class_time = check_gene

                    elif isinstance(check_gene, Room):
                        check_room = check_gene

                    elif isinstance(check_gene, Course):
                        check_instructor = None
                        check_class_time = None
                        check_room = None

                        one_class_one_time = True
                        consecutive_time_slots = False
                        far_away_rooms = False

                        check_course = check_gene

                    if check_course and course == check_course:
                        continue

                    if check_class_time and check_instructor and check_room and check_course:
                        if (
                            check_instructor == instructor
                            and check_class_time == class_time
                        ):
                            one_class_one_time = False
                        
                        if check_class_time.start - class_time.start == timedelta(hours=1):
                            consecutive_time_slots = True

                            # check if we're in a separate remote building
                            if (
                                room.building in remote_buildings
                                and room.building != check_room.building
                                and check_room.building in remote_buildings
                            ):
                                far_away_rooms = True
    
                        check_course = None

                ## consecutive time slots
                if consecutive_time_slots:

                    ## far away rooms
                    if far_away_rooms:
                        chromosome.checks.instructor_consecutive_slots_far_away_rooms -= 0.4
                        
                    else:
                        chromosome.checks.instructor_consecutive_slots += 0.5

                ## one class one time
                if one_class_one_time:
                    chromosome.checks.instructor_one_class_one_time += 0.2

                ## multiple class same time
                else:
                    # I THINK by deduction, this makes sense
                    chromosome.checks.instructor_multiple_classes_one_time -= 0.2

                course = None

        for instructor, count in instructor_counts.items():
            if count > 4:
                chromosome.checks.instructor_more_than_4_classes -= 0.5

            elif count < 3:
                if instructor != "Xu":
                    chromosome.checks.instructor_less_than_3_classes -= 0.4

    def course_specific_check() -> None:
        """
        - The 2 sections of CS 101 are more than 4 hours apart: +0.5
        - Both sections of CS 101 are in the same time slot: -0.5
        - The 2 sections of CS 191 are more than 4 hours apart: +0.5
        - Both sections of CS 191 are in the same time slot: -0.5
        - A section of CS 191 and a section of CS 101 are taught in 
            consecutive time slots: +0.5
        - If consective time slots and one class is in remote building Bloch or 
        Katz and the following class in the other remote building: -0.4
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
                class_time = None
                room = None

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
                    if isinstance(check_gene, Course):
                        check_class_time = None
                        check_room = None

                        consecutive_time_slots = False
                        one_hour_gap = False
                        same_time = False
                        far_away_rooms = False
                        four_hour_gap = False

                        check_course = check_gene

                    elif isinstance(check_gene, ClassTime):
                        check_class_time = check_gene

                    elif isinstance(check_gene, Room):
                        check_room = check_gene

                    if check_course and course == check_course:
                        continue

                    if check_course and check_class_time and check_room:

                        # These are aribitrary checks for any given 2 courses.
                        if check_class_time.start - class_time.start == timedelta(hours=0):
                            same_time = True
                        elif check_class_time.start - class_time.start == timedelta(hours=1):
                            consecutive_time_slots = True

                            # check if we're in a separate remote building
                            if (
                                room.building in remote_buildings
                                and room.building != check_room.building
                                and check_room.building in remote_buildings
                            ):
                                far_away_rooms = True

                        elif check_class_time.start - class_time.start == timedelta(hours=2):
                            one_hour_gap = True

                        elif check_class_time.start - class_time.start == timedelta(hours=4):
                            four_hour_gap = True

                        # section following another
                        if (
                            "CS101" in check_course.name and "CS191" in course.name
                            or "CS191" in check_course.name and "CS101" in course.name
                        ):
                            if consecutive_time_slots:
                                if far_away_rooms:
                                    chromosome.checks.sections_consecutive_far_away_rooms -= 0.4
                                    
                                chromosome.checks.cs_101_191_consecutive += 0.5

                            elif one_hour_gap:
                                chromosome.checks.cs_101_191_one_hour_apart += 0.25

                            elif same_time:
                                chromosome.checks.cs_101_191_same_time -= 0.25

                        # comparing same class different sections
                        elif (
                            "CS101A" == course.name and "CS101B" == check_course.name
                            or "CS191A" == course.name and "CS191B" == check_course.name
                        ):
                            if same_time:
                                if "CS101" in check_course.name:
                                    chromosome.checks.cs_101_same_time = -0.5
                                elif "CS191" in check_course.name:
                                    chromosome.checks.cs_191_same_time = -0.5

                            elif four_hour_gap:
                                if "CS101" in check_course.name:
                                    chromosome.checks.cs_101_4_hours_apart = 0.5
                                elif "CS191" in check_course.name:
                                    chromosome.checks.cs_191_4_hours_apart = 0.5

                        check_course = None

                course = None

        chromosome.checks.cs_101_191_same_time /= 2
        chromosome.checks.cs_101_191_one_hour_apart /= 2

    chromosome.checks = chromosome.Checks()
    same_room_same_time_check()
    room_size_check()
    preferred_instructor_check()
    instructor_load_check()
    course_specific_check()
    chromosome.set_fitness_from_checks()

    if print_checks:
        for check in fields(chromosome.checks):
            print(f"    {check.name}: {getattr(chromosome.checks, check.name):.2f}")
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
# Write Chromosome Checks to File

def write_chromosome_checks_to_file(
    chromosome: Chromosome, file: TextIOWrapper) -> None:
    file.write(
        f"{chromosome.fitness:.3f},"
        + ','.join([f'{c:.3f}' for c in chromosome.checks.__dict__.values()])
        + '\n'
    )

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Save Solution

def save_solution(chromosome: Chromosome) -> None:
    with open("Solution/Courses.txt", "w+") as f:
        for gene in chromosome.genes:
            if isinstance(gene, Course):
                info = gene
                f.write(f"\n\nCourse: {info.name}\n")

            elif isinstance(gene, ClassTime):
                class_time = gene
                f.write(f"Class Time: {class_time.start.strftime('%I:%M %p')} - {class_time.end.strftime('%I:%M %p')}\n")
            
            elif isinstance(gene, Instructor):
                instructor = gene
                f.write(f"Instructor: {instructor.name}\n")
            
            elif isinstance(gene, Room):
                room = gene
                f.write(f"Room: {gene.building} ({gene.room}) {info.expected_enrollment}/{room.capacity} ({info.expected_enrollment / room.capacity * 100:.2f}% full)\n")

    with open("Solution/Checks.txt", "w+") as f:
        for check in fields(chromosome.checks):
            f.write(f"{check.name}: {getattr(chromosome.checks, check.name):.2f}\n")
        f.write(f"Fitness: {chromosome.fitness:.2f}")

    with open("Solution/Instructors.txt", "w+") as f:
        @dataclass
        class IntstructorInfo:
            name: str
            course: Course
            class_time: ClassTime
            room: Room

        course: Course = None
        instructor: Instructor = None
        room: Room = None
        class_time: ClassTime = None
        instructors: dict[str, list[IntstructorInfo]] = {}
        for gene in chromosome.genes:
            if isinstance(gene, Course):
                instructor = None
                room = None
                class_time = None

                course = gene

            elif isinstance(gene, ClassTime):
                class_time = gene

            elif isinstance(gene, Instructor):
                instructor = gene

            elif isinstance(gene, Room):
                room = gene
                
            if course and instructor and room and class_time:
                if instructor.name not in instructors:
                    instructors[instructor.name] = []
                
                instructors[instructor.name].append(
                    IntstructorInfo(
                        name=instructor.name,
                        course=course,
                        class_time=class_time,
                        room=room
                    )
                )
                
        for instructor in instructors:
            instructors[instructor].sort(key=lambda x: x.class_time.start)
            f.write(f"\nInstructor: {instructor}\n")
            for info in instructors[instructor]:
                f.write(f"    Course: {info.course.name}\n")
                f.write(f"    Class Time: {info.class_time.start.strftime('%I:%M %p')} - {info.class_time.end.strftime('%I:%M %p')}\n")
                f.write(f"    Room: {info.room.building} ({info.room.room}) {info.course.expected_enrollment}/{info.room.capacity} ({info.course.expected_enrollment / info.room.capacity * 100:.2f}% full)\n\n")

    with open("Solution/Rooms.txt", "w+") as f:
        @dataclass
        class RoomInfo:
            course: Course
            room: Room
            class_time: ClassTime
            instructor: Instructor

        course: Course = None
        instructor: Instructor = None
        room: Room = None
        class_time: ClassTime = None
        rooms: dict[str, list[RoomInfo]] = {}
        for gene in chromosome.genes:
            if isinstance(gene, Course):
                instructor = None
                room = None
                class_time = None

                course = gene

            elif isinstance(gene, ClassTime):
                class_time = gene

            elif isinstance(gene, Instructor):
                instructor = gene

            elif isinstance(gene, Room):
                room = gene
                
            if course and instructor and room and class_time:
                if room.key not in rooms:
                    rooms[room.key] = []
                
                rooms[room.key].append(
                    RoomInfo(
                        course=course,
                        room=room,
                        class_time=class_time,
                        instructor=instructor
                    )
                )
                
        for courses in rooms.values():
            courses.sort(key=lambda x: x.class_time.start)
            f.write(f"\nRoom: {courses[0].room.building} ({courses[0].room.room})\n")
            for info in courses:
                f.write(f"    Class Time: {info.class_time.start.strftime('%I:%M %p')} - {info.class_time.end.strftime('%I:%M %p')}\n")
                f.write(f"    Course: {info.course.name}\n")
                f.write(f"    Instructor: {info.instructor.name}\n")
                f.write(f"    Capacity: {info.course.expected_enrollment}/{info.room.capacity} ({info.course.expected_enrollment / info.room.capacity * 100:.2f}% full)\n\n")

    with open("Solution/TimeSlots.txt", "w+") as f:
        @dataclass
        class TimeSlotInfo:
            class_time: ClassTime
            course: Course
            instructor: Instructor
            room: Room

        course: Course = None
        instructor: Instructor = None
        room: Room = None
        class_time: ClassTime = None
        time_slots: dict[str, list[TimeSlotInfo]] = {}
        for gene in chromosome.genes:
            if isinstance(gene, Course):
                instructor = None
                room = None
                class_time = None

                course = gene

            elif isinstance(gene, ClassTime):
                class_time = gene

            elif isinstance(gene, Instructor):
                instructor = gene

            elif isinstance(gene, Room):
                room = gene
                
            if course and instructor and room and class_time:
                if class_time.start not in time_slots:
                    time_slots[class_time.start] = []
                
                time_slots[class_time.start].append(
                    TimeSlotInfo(
                        class_time=class_time,
                        course=course,
                        instructor=instructor,
                        room=room
                    )
                )
                
        time_slots = {k: time_slots[k] for k in sorted(time_slots)}
        for time_slot in time_slots:
            time_slots[time_slot].sort(key=lambda x: x.course.name)
            f.write(f"\nTime Slot: {time_slots[time_slot][0].class_time.start.strftime('%I:%M %p')} - {time_slots[time_slot][0].class_time.end.strftime('%I:%M %p')}\n")
            for info in time_slots[time_slot]:
                f.write(f"    Course: {info.course.name}\n")
                f.write(f"    Instructor: {info.instructor.name}\n")
                f.write(f"    Room: {info.room.building} ({info.room.room}) {info.course.expected_enrollment}/{info.room.capacity} ({info.course.expected_enrollment / info.room.capacity * 100:.2f}% full)\n\n")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# main()

def main(args: list[str]) -> None:
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

    tests(possible_genes=possible_genes)

    # initialize the ga
    scheduling_ga: GeneticAlgorithm = GeneticAlgorithm(
        population_size=1000,
        mutation_rate=0.01,
        possible_genes=possible_genes,
        chromosome_generator=generate_chromosomes,
        chromosome_evaluator=evaluate_chromosome,
        chromosome_displayer=display_chromosome,
        chromosome_mutator=mutate_chromosome,
        chromosome_writer=save_solution,
    )
    scheduling_ga.initialize_population(count=scheduling_ga.population_size)

    # setup plot
    fig = plt.figure()
    fitness_subplot = fig.add_subplot(1, 1, 1)
    standard_deviation_subplot = fitness_subplot.twinx()
    fitness_subplot.set_xlabel("Generation")
    fitness_subplot.set_ylabel("Fitness")
    standard_deviation_subplot.set_ylabel("Standard Deviation", rotation=270, labelpad=15)
    fitness_subplot.legend(
        handles=[
            mpatches.Patch(color="green", label="Fittest Chromosome"),
            mpatches.Patch(color="blue", label="Average Fitness"),
            mpatches.Patch(color="red", label="Standard Deviation")
        ]
    )

    # run the ga
    minimum_improvement: float = 0.01
    maximum_generations: int = 100
    generations: int = 1
    print(f"Starting Genetic Algorithm")
    print(f"Population Size: {scheduling_ga.population_size}")
    print(f"Mutation Rate: {scheduling_ga.mutation_rate}")
    print(f"Maximum Generations: {maximum_generations}")
    print(f"Minimum Improvement: {minimum_improvement}")

    improvement: float = float('inf')
    previous_average_fitness: float = 0
    while True:
        # Evaluate and reproduce
        print(f"\nGeneration {generations}")
        scheduling_ga.evaluate_chromosomes()

        ## calculate improvement
        if previous_average_fitness == 0:
            improvement = 1
        else:
            improvement = (
                (scheduling_ga.average_fitness - previous_average_fitness) 
                / previous_average_fitness
            )
        previous_average_fitness = scheduling_ga.average_fitness

        print(f"Fittest Chromosome: {scheduling_ga.fittest_chromosome.fitness:.2f}")
        print(f"Average Fitness: {scheduling_ga.average_fitness:.2f}")
        print(f"Standard Deviation: {scheduling_ga.standard_deviation:.2f}")
        print(f"Improvement: {improvement * 100:.2f}%")

        ## plot the data
        # plot box plot
        # average_fitness_subplot.boxplot(scheduling_ga.fitnesses, positions=[i])

        fitness_subplot.plot(generations, scheduling_ga.fittest_chromosome.fitness, "go")
        fitness_subplot.plot(generations, scheduling_ga.average_fitness, "bo")
        standard_deviation_subplot.plot(generations, scheduling_ga.standard_deviation, "ro")

        # break condition
        if minimum_improvement > improvement and generations >= maximum_generations:
            break

        scheduling_ga.calculate_probabilities()
        scheduling_ga.create_offspring()

        generations += 1

    fittest_chromosome: Chromosome = scheduling_ga.fittest_chromosome
    scheduling_ga.display_chromosome(fittest_chromosome)
    print("\nChecks:")
    evaluate_chromosome(fittest_chromosome, print_checks=True)
    scheduling_ga.save_chromosome(fittest_chromosome)

    plt.show()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def tests(possible_genes: dict[type, list[Gene]]) -> None:

    # set random seed
    rnd.seed(0)

    # same room same time check
    # all courses are set at the same time in the same room
    same_room_same_time_check: Chromosome = generate_chromosomes(possible_genes, 1)[0]
    for gene in same_room_same_time_check.genes:
        if isinstance(gene, ClassTime):
            gene.start = possible_genes[ClassTime][0].start
            gene.end = possible_genes[ClassTime][0].end

        elif isinstance(gene, Room):
            gene.building = possible_genes[Room][0].building
            gene.room = possible_genes[Room][0].room

    same_room_same_time_check = evaluate_chromosome(same_room_same_time_check)
    assert same_room_same_time_check.checks.same_room_same_time == -0.5 * (len(possible_genes[Course]) - 1)

    # room size checks
    room_size_checks: Chromosome = generate_chromosomes(possible_genes, 1)[0]
    for i, gene in enumerate(room_size_checks.genes):
        if isinstance(gene, Room):
            room_size_checks.genes[i] = possible_genes[Room][0]

    room_size_checks = evaluate_chromosome(room_size_checks)
    assert room_size_checks.checks.room_too_small == -0.5 * 1
    assert room_size_checks.checks.room_3x_too_big == -0.2 * 1
    assert room_size_checks.checks.room_6x_too_big == 0
    assert room_size_checks.checks.room_size_sufficient == 0.3 * 9

    # preferred instructor checks
    preferred_instructor_checks: Chromosome = generate_chromosomes(possible_genes, 1)[0]
    for i, gene in enumerate(preferred_instructor_checks.genes):
        if isinstance(gene, Instructor):
            preferred_instructor_checks.genes[i] = possible_genes[Instructor][0]

    preferred_instructor_checks = evaluate_chromosome(preferred_instructor_checks)
    assert preferred_instructor_checks.checks.preferred_instructor == 0.5 * 5
    assert preferred_instructor_checks.checks.other_instructor == 0.2 * 0
    assert preferred_instructor_checks.checks.other_faculty == round(-0.1 * 6, 2)

    # instructor load checks
    instructor_load_checks: Chromosome = generate_chromosomes(possible_genes, 1)[0]

    instructor_load_checks = evaluate_chromosome(instructor_load_checks)
    assert round(instructor_load_checks.checks.instructor_one_class_one_time, 2) == 0.2 * 11
    assert instructor_load_checks.checks.instructor_multiple_classes_one_time == -0.2 * 0
    assert instructor_load_checks.checks.instructor_more_than_4_classes == -0.5 * 0
    assert round(instructor_load_checks.checks.instructor_less_than_3_classes, 2) == -0.4 * 5
    assert instructor_load_checks.checks.instructor_consecutive_slots_far_away_rooms == -0.4 * 0

    # course specific checks
    course_specific_checks: Chromosome = instructor_load_checks        
    
    display_chromosome(course_specific_checks)
    course_specific_checks = evaluate_chromosome(course_specific_checks)
    assert course_specific_checks.checks.cs_101_4_hours_apart == 0.5 * 0
    assert course_specific_checks.checks.cs_101_same_time == -0.5 * 0
    assert course_specific_checks.checks.cs_191_4_hours_apart == 0.5 * 0
    assert course_specific_checks.checks.cs_191_same_time == -0.5 * 0
    assert course_specific_checks.checks.cs_101_191_consecutive == 0.5 * 1
    assert course_specific_checks.checks.sections_consecutive_far_away_rooms == -0.4 * 0
    assert course_specific_checks.checks.cs_101_191_one_hour_apart == 0.25 * 1
    assert course_specific_checks.checks.cs_101_191_same_time == -0.25 * 0

    rnd.seed(datetime.now().microsecond)

if __name__ == "__main__":
    args: list[str] = sys.argv[1:]
    main(args)