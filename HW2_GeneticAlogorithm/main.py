# Chris Seagraves
# Course Scheduling with a Genetic Algorithm

# python imports
from copy import deepcopy
from dataclasses import dataclass, fields
from datetime import datetime, timedelta
from io import TextIOWrapper
from itertools import combinations
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import pickle
import random as rnd
from random import choice, random
import sys


# GeneticAlgorithm imports
# from GeneticAlgorithm.Chromosome import Chromosome
from GeneticAlgorithm.Gene import Gene
from GeneticAlgorithm.GeneticAlgorithm import GeneticAlgorithm

# support imports
from ClassTime import ClassTime
from Chromosome import Check, Checks, Chromosome
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

    @dataclass
    class CourseGene:
        course: Course = None
        class_time: ClassTime = None
        instructor: Instructor = None
        room: Room = None

    chromosome.checks = Checks()

    remote_buildings: set[str] = {"Bloch", "Katz"}
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # split chromosome into courses

    chromosome_by_courses: dict[str, CourseGene] = {}
    chromosome_by_class_time: dict[datetime, list[CourseGene]] = {}
    chromosome_by_instructor: dict[str, list[CourseGene]] = {}
    chromosome_by_room: dict[str, list[CourseGene]] = {}

    course_gene: CourseGene = CourseGene()
    for gene in chromosome.genes:
        if isinstance(gene, Course):
            course_gene.course=gene
        elif isinstance(gene, ClassTime):
            course_gene.class_time = gene
        elif isinstance(gene, Instructor):
            course_gene.instructor = gene
        elif isinstance(gene, Room):
            course_gene.room = gene

        if (course_gene.course 
            and course_gene.class_time
            and course_gene.instructor
            and course_gene.room):

            # add to courses
            chromosome_by_courses[course_gene.course.name] = course_gene

            # add to class times
            if course_gene.class_time.start not in chromosome_by_class_time:
                chromosome_by_class_time[course_gene.class_time.start] = []
            chromosome_by_class_time[course_gene.class_time.start].append(course_gene)

            # add to instructors
            if course_gene.instructor.name not in chromosome_by_instructor:
                chromosome_by_instructor[course_gene.instructor.name] = []
            chromosome_by_instructor[course_gene.instructor.name].append(course_gene)

            # add to rooms
            if course_gene.room.key not in chromosome_by_room:
                chromosome_by_room[course_gene.room.key] = []
            chromosome_by_room[course_gene.room.key].append(course_gene)

            course_gene = CourseGene()
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # Check for room sizes and multiple bookings.

    for room_key in chromosome_by_room:
        chromosome_by_room[room_key].sort(
            key=lambda c: c.class_time.start)

        for i, course in enumerate(chromosome_by_room[room_key]):

            # room too small
            if course.room.capacity < course.course.expected_enrollment:
                chromosome.checks.room_too_small.count += 1

            # room 6x too big
            elif course.room.capacity >= course.course.expected_enrollment * 6:
                chromosome.checks.room_6x_too_big.count += 1

            # room 3x too big
            elif course.room.capacity >= course.course.expected_enrollment * 3:
                chromosome.checks.room_3x_too_big.count += 1

            # room sufficient
            else:
                chromosome.checks.room_size_sufficient.count += 1

            if i == 0:
                continue

            previous_course: CourseGene = chromosome_by_room[room_key][i-1]

            # multiple courses in same room at the same time
            if course.class_time.start == previous_course.class_time.start:
                chromosome.checks.same_room_same_time.count += 1

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # Check for who's teaching the courses.

    for course in chromosome_by_courses.values():
        if course.instructor.name in course.course.preferred_instructors:
            chromosome.checks.preferred_instructor.count += 1

        elif course.instructor.name in course.course.other_instructors:
            chromosome.checks.other_instructor.count += 1

        else:
            chromosome.checks.other_faculty.count += 1

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # Check for specific courses and sections
    
    cs101A = chromosome_by_courses[f"CS101A"]
    cs101B = chromosome_by_courses[f"CS101B"]
    cs191A = chromosome_by_courses[f"CS191A"]
    cs191B = chromosome_by_courses[f"CS191B"]

    # order independent combinations
    section_combinations = combinations([cs101A, cs101B, cs191A, cs191B], 2)

    cs101_delta = cs101A.class_time.start - cs101B.class_time.start
    cs191_delta = cs191A.class_time.start - cs191B.class_time.start

    if abs(cs101_delta) == timedelta(hours=0):
        chromosome.checks.cs_101_same_time.count += 1

    elif abs(cs101_delta) > timedelta(hours=4):
        chromosome.checks.cs_101_4_hour_gap.count += 1
 
    if abs(cs191_delta) == timedelta(hours=0):
        chromosome.checks.cs_191_same_time.count += 1
    
    elif abs(cs191_delta) > timedelta(hours=4):
        chromosome.checks.cs_191_4_hour_gap.count += 1

    for section_1, section_2 in section_combinations:
        # section if following another section (not of same class)
        if (section_1 in [cs101A, cs101B] 
            and section_2 in [cs191A, cs191B]):
            class_delta = section_1.class_time.start - section_2.class_time.start

            # same time
            if abs(class_delta) == timedelta(hours=0):
                chromosome.checks.cs_101_191_same_time.count += 1
            
            # consecutive
            elif abs(class_delta) == timedelta(hours=1):
                chromosome.checks.cs_101_191_consecutive.count += 1

                # consecutive seperate remote buildings
                if (section_1.room.building in remote_buildings 
                    and section_1.room.building != section_2.room.building
                    and section_2.room.building in remote_buildings):
                    chromosome.checks.sections_consecutive_far_away_rooms.count += 1

            # 1 hour gap
            elif abs(class_delta) == timedelta(hours=2):
                chromosome.checks.cs_101_191_one_hour_gap.count += 1

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # Check for instructor load.

    for instructor_name in chromosome_by_instructor:
        chromosome_by_instructor[instructor_name].sort(
            key=lambda c: c.class_time.start)

        if len(chromosome_by_instructor[instructor_name]) > 4:
            chromosome.checks.instructor_more_than_4_classes.count += 1

        elif 0 < len(chromosome_by_instructor[instructor_name]) < 3:
            if instructor_name != "Xu":
                chromosome.checks.instructor_less_than_3_classes.count += 1

        for i, course in enumerate(chromosome_by_instructor[instructor_name]):
            chromosome.checks.instructor_one_class_one_time.count += 1

            if i == 0:
                continue

            previous_course: CourseGene = chromosome_by_instructor[instructor_name][i-1]

            # same room same time
            if course.class_time.start == previous_course.class_time.start:
                chromosome.checks.instructor_multiple_classes_one_time.count += 1
                chromosome.checks.instructor_one_class_one_time.count -= 1
            
            # consecutive time slots
            if course.class_time.start - previous_course.class_time.start == timedelta(hours=1):
                chromosome.checks.instructor_consecutive_slots.count += 1

                # consecutive separate remote buildings
                if (course.room.building in remote_buildings
                    and course.room.building != previous_course.room.building
                    and previous_course.room.building in remote_buildings):
                    chromosome.checks.instructor_consecutive_slots_far_away_rooms.count += 1

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    chromosome.set_fitness_from_checks()

    if print_checks:
        chromosome.checks.print_checks()
        print('-' * 40)
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
            print(f"\nCourse: {course.name}")

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
                f.write(f"\nCourse: {info.name}  \n")

            elif isinstance(gene, ClassTime):
                class_time = gene
                f.write(f"Class Time: {class_time.start.strftime('%I:%M %p')} - {class_time.end.strftime('%I:%M %p')}  \n")
            
            elif isinstance(gene, Instructor):
                instructor = gene
                f.write(f"Instructor: {instructor.name}  \n")
            
            elif isinstance(gene, Room):
                room = gene
                f.write(f"Room: {gene.building} ({gene.room}) {info.expected_enrollment}/{room.capacity} ({info.expected_enrollment / room.capacity * 100:.2f}% full)  \n")

    with open("Solution/Checks.txt", "w+") as f:
        for check_name, check in chromosome.checks.__dict__.items():
            check: Check = check
            f.write(f"{check_name}: {check.count} * {check.weight} = {check.score}  \n")
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
            f.write(f"\nInstructor: {instructor}  \n")
            for info in instructors[instructor]:
                f.write(f"    Course: {info.course.name}  \n")
                f.write(f"    Class Time: {info.class_time.start.strftime('%I:%M %p')} - {info.class_time.end.strftime('%I:%M %p')}  \n")
                f.write(f"    Room: {info.room.building} ({info.room.room}) {info.course.expected_enrollment}/{info.room.capacity} ({info.course.expected_enrollment / info.room.capacity * 100:.2f}% full)  \n")

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
            f.write(f"\nRoom: {courses[0].room.building} ({courses[0].room.room})  \n")
            for info in courses:
                f.write(f"    Class Time: {info.class_time.start.strftime('%I:%M %p')} - {info.class_time.end.strftime('%I:%M %p')}  \n")
                f.write(f"    Course: {info.course.name}\n")
                f.write(f"    Instructor: {info.instructor.name}  \n")
                f.write(f"    Capacity: {info.course.expected_enrollment}/{info.room.capacity} ({info.course.expected_enrollment / info.room.capacity * 100:.2f}% full)  \n\n")

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
            f.write(f"\nTime Slot: {time_slots[time_slot][0].class_time.start.strftime('%I:%M %p')} - {time_slots[time_slot][0].class_time.end.strftime('%I:%M %p')}  \n")
            for info in time_slots[time_slot]:
                f.write(f"    Course: {info.course.name}  \n")
                f.write(f"    Instructor: {info.instructor.name}  \n")
                f.write(f"    Room: {info.room.building} ({info.room.room}) {info.course.expected_enrollment}/{info.room.capacity} ({info.course.expected_enrollment / info.room.capacity * 100:.2f}% full)  \n\n")

    with open("Solution/Chromosome.pickle", "wb+") as f:
        pickle.dump(chromosome, f)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# main()

def main(args: list[str]) -> None:

    # read the data
    print("Reading data...")
    courses: pd.DataFrame = pd.read_csv("Database/courses.csv", dtype=str)
    class_times: pd.DataFrame = pd.read_csv("Database/class_times.csv", dtype=str)
    rooms: pd.DataFrame = pd.read_csv("Database/rooms.csv", dtype=str)
    faculty: pd.DataFrame = pd.read_csv("Database/faculty.csv", dtype=str)

    # create the objects
    print("Creating dictionary of possible genes...")
    possible_genes: dict[type, list[Gene]] = {
        Course: create_courses(courses),
        ClassTime: create_class_times(class_times),
        Room: create_rooms(rooms),
        Instructor: create_instructors(faculty)
    }

    # Assertion tests
    print("Checking assertions...")
    tests(possible_genes=possible_genes)

    # initialize the ga
    print("Initializing the GA...")
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
    print("Setting up the plot...")
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
    maximum_generations: int = 150
    print(f"\nStarting GA...")
    print(f"Population Size: {scheduling_ga.population_size}")
    print(f"Mutation Rate: {scheduling_ga.mutation_rate}")
    print(f"Maximum Generations: {maximum_generations}")
    print(f"Minimum Improvement: {minimum_improvement}")

    improvement: float = float('inf')
    previous_average_fitness: float = 0
    generations: int = 1
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
        if minimum_improvement > abs(improvement) and generations >= maximum_generations:
            break

        scheduling_ga.calculate_probabilities()
        scheduling_ga.create_offspring()

        generations += 1

    fittest_chromosome: Chromosome = scheduling_ga.fittest_chromosome
    print('Saving solution...')
    scheduling_ga.save_chromosome(fittest_chromosome)

    scheduling_ga.display_chromosome(fittest_chromosome)
    print("\nChecks:")
    evaluate_chromosome(fittest_chromosome, print_checks=True)

    plt.show()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def tests(possible_genes: dict[type, list[Gene]]) -> None:

    # previous solution's chromosome
    # with open("Solution/Chromosome.pickle", "rb") as file:
    #     previous_solution: Chromosome = pickle.load(file)
    # display_chromosome(previous_solution)
    # previous_solution = evaluate_chromosome(previous_solution, print_checks=True)

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
    assert same_room_same_time_check.checks.same_room_same_time.count == len(possible_genes[Course]) - 1

    # room size checks
    room_size_checks: Chromosome = generate_chromosomes(possible_genes, 1)[0]
    for i, gene in enumerate(room_size_checks.genes):
        if isinstance(gene, Room):
            room_size_checks.genes[i] = possible_genes[Room][0]
            
    room_size_checks = evaluate_chromosome(room_size_checks)
    assert room_size_checks.checks.room_too_small.count == 1
    assert room_size_checks.checks.room_3x_too_big.count == 1
    assert room_size_checks.checks.room_6x_too_big.count == 0
    assert room_size_checks.checks.room_size_sufficient.count == 9

    # preferred instructor checks
    preferred_instructor_checks: Chromosome = generate_chromosomes(possible_genes, 1)[0]
    for i, gene in enumerate(preferred_instructor_checks.genes):
        if isinstance(gene, Instructor):
            preferred_instructor_checks.genes[i] = possible_genes[Instructor][0]

    preferred_instructor_checks = evaluate_chromosome(preferred_instructor_checks)
    assert preferred_instructor_checks.checks.preferred_instructor.count == 5
    assert preferred_instructor_checks.checks.other_instructor.count == 0
    assert preferred_instructor_checks.checks.other_faculty.count == 6

    # instructor load checks
    instructor_load_checks: Chromosome = generate_chromosomes(possible_genes, 1)[0]

    instructor_load_checks = evaluate_chromosome(instructor_load_checks)
    assert instructor_load_checks.checks.instructor_one_class_one_time.count == 11
    assert instructor_load_checks.checks.instructor_multiple_classes_one_time.count == 0
    assert instructor_load_checks.checks.instructor_more_than_4_classes.count == 0
    assert instructor_load_checks.checks.instructor_less_than_3_classes.count == 5
    assert instructor_load_checks.checks.instructor_consecutive_slots_far_away_rooms.count == 0

    # course specific checks
    course_specific_checks: Chromosome = instructor_load_checks        
    
    course_specific_checks = evaluate_chromosome(course_specific_checks)
    assert course_specific_checks.checks.cs_101_4_hour_gap.count == 0
    assert course_specific_checks.checks.cs_101_same_time.count == 0
    assert course_specific_checks.checks.cs_191_4_hour_gap.count == 0
    assert course_specific_checks.checks.cs_191_same_time.count == 0
    assert course_specific_checks.checks.cs_101_191_consecutive.count == 1
    assert course_specific_checks.checks.sections_consecutive_far_away_rooms.count == 0
    assert course_specific_checks.checks.cs_101_191_one_hour_gap.count == 2
    assert course_specific_checks.checks.cs_101_191_same_time.count == 0

    rnd.seed(datetime.now().microsecond)

if __name__ == "__main__":
    args: list[str] = sys.argv[1:]
    main(args)