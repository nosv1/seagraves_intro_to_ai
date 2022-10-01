# Chris Seagraves
# Course Scheduling with a Genetic Algorithm

# python imports
from datetime import datetime, timedelta
import pandas as pd
from random import choice

# GeneticAlgorithm imports
from GeneticAlgorithm.Chromosome import Chromosome
from GeneticAlgorithm.Gene import Gene
from GeneticAlgorithm.GeneticAlgorithm import GeneticAlgorithm

# support imports
from ClassTime import ClassTime
from Course import Course
from Instructor import Instructor
from Room import Room

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

def generate_chromosomes(
    courses: list[Course], 
    class_times: list[ClassTime], 
    rooms: list[Room],
    instructors: list[Instructor],
    count: int=1
) -> list[Chromosome]:
    chromosomes: list[Chromosome] = []
    for i in range(count):
        chromosome: Chromosome = Chromosome(genes=[])
        for course in courses:
            chromosome.genes.append(course)
            chromosome.genes.append(choice(rooms))
            chromosome.genes.append(choice(class_times))
            chromosome.genes.append(choice(instructors))
        chromosomes.append(chromosome)
    return chromosomes

def evaluate_chromosome(chromosome: Chromosome) -> float:

    """
    The idea with these checks is to loop through the genes, storing the 
    values we care about until we get to the next course, then evaluating the 
    stored values.
    We KNOW Course goes first, but the order of the following genes isn't a given.
    """

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
                        chromosome.fitness -= 0.5
    
    def room_size_check() -> None:
        """
        Room size:
        - Class is in a room too small for its expected enrollment: -0.5
        - Class is in a room with capacity > 3 times expected enrollment: -0.2
        - Class is in a room with capacity > 6 times expected enrollment: -0.4
        - Otherwise +0.3
        """
        course: Course = None
        for gene in chromosome.genes:
            if isinstance(gene, Course):
                course = gene

            elif isinstance(gene, Room):
                room = gene

                if room.capacity < course.expected_enrollment:
                    chromosome.fitness -= 0.5

                elif room.capacity >= course.expected_enrollment * 3:
                    chromosome.fitness -= 0.2

                elif room.capacity >= course.expected_enrollment * 6:
                    chromosome.fitness -= 0.4

                else:
                    chromosome.fitness += 0.3

    def preferred_instructor_check() -> None:
        """
        Class is taught by a preferred faculty member: +0.5
        Class is taught by another faculty member listed for that course: +0.2
        Class is taught by some other faculty: -0.1
        """
        course: Course = None
        for gene in chromosome.genes:
            if isinstance(gene, Course):
                course = gene

            elif isinstance(gene, Instructor):
                instructor = gene

                if instructor.name in course.preferred_instructors:
                    chromosome.fitness += 0.5

                elif instructor.name in course.other_instructors:
                    chromosome.fitness += 0.2

                else:
                    chromosome.fitness -= 0.1
        
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

            if instructor and class_time:
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
                        course_count += 1 

                    elif isinstance(check_gene, ClassTime):
                        check_class_time = check_gene

                    elif isinstance(check_gene, Room):
                        check_room = check_gene

                    elif isinstance(check_gene, Course):
                        check_course = check_gene

                    if check_instructor and check_class_time:
                        if (
                            check_instructor == instructor and 
                            check_class_time == class_time
                        ):
                            one_class_one_time = False
                        
                        if check_class_time.start - class_time.start == timedelta(hours=1):
                            consecutive_time_slots = True

                            if (
                                room.building in ["Bloch", "Katz"] and
                                check_room.building not in ["Bloch", "Katz"]
                            ):
                                far_away_rooms = True

                ## course count
                if course_count > 4:
                    chromosome.fitness -= 0.5

                elif course_count < 3:
                    if instructor.name != "Xu":
                        chromosome.fitness -= 0.4

                ## one class one time
                if one_class_one_time:
                    chromosome.fitness += 0.2
                else:
                    # I THINK by deduction, this makes sense
                    chromosome.fitness -= 0.2

                ## consecutive time slots
                if consecutive_time_slots:
                    if far_away_rooms:
                        chromosome.fitness -= 0.4
                    else:
                        chromosome.fitness += 0.5

                # set to None so we don't the loop again for the same course
                instructor = None
                class_time = None

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
                                room.building in ["Bloch", "Katz"] and
                                check_room.building not in ["Bloch", "Katz"]
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
                                    chromosome.fitness -= 0.4
                                else:
                                    chromosome.fitness += 0.5

                            elif one_hour_gap:
                                chromosome.fitness += 0.25

                            elif same_time:
                                chromosome.fitness -= 0.25

                        # both sections
                        elif (
                            "CS101" in course.name and "CS101" in check_course.name or
                            "CS191" in course.name and "CS191" in check_course.name
                        ):
                            if same_time:
                                chromosome.fitness -= 0.5

                            elif four_hour_gap:
                                chromosome.fitness += 0.5

                # set to None so we don't the loop again for the same course
                course = None
                class_time = None
                room = None     

    same_room_same_time_check()
    room_size_check()
    preferred_instructor_check()
    instructor_load_check()
    course_specific_check()

    return chromosome

def display_chromosome(chromosome: Chromosome) -> None:
    """
    Displays a chromosome in a readable format.
    """
    for gene in chromosome.genes:
        if isinstance(gene, Course):
            print(f"\n\nCourse: {gene.name}")

        elif isinstance(gene, ClassTime):
            print(f"Class Time: {gene.start.strftime('%I:%M %p')} - {gene.end.strftime('%I:%M %p')}")
        
        elif isinstance(gene, Instructor):
            print(f"Instructor: {gene.name}")
        
        elif isinstance(gene, Room):
            print(f"Room: {gene.building} ({gene.room})")
    print("-" * 40)
    print(f"Fitness: {chromosome.fitness:.2f}")

def main() -> None:
    courses: pd.DataFrame = pd.read_csv("Database/courses.csv", dtype=str)
    class_times: pd.DataFrame = pd.read_csv("Database/class_times.csv", dtype=str)
    rooms: pd.DataFrame = pd.read_csv("Database/rooms.csv", dtype=str)
    faculty: pd.DataFrame = pd.read_csv("Database/faculty.csv", dtype=str)

    courses: list[Course] = create_courses(courses)
    class_times: list[ClassTime] = create_class_times(class_times)
    rooms: list[Room] = create_rooms(rooms)
    instructors: list[Instructor] = create_instructors(faculty)

    scheduling_ga: GeneticAlgorithm = GeneticAlgorithm(
        population_size=10,
        mutation_rate=0.1,
        chromosome_generator=generate_chromosomes,
        chromosome_evaluator=evaluate_chromosome,
        chromosome_displayer=display_chromosome,
    )

    """
    We lose time converting values to list, but we save time indexing the dict 
    in the fitness function.
    """
    scheduling_ga.initialize_population(
        count=scheduling_ga.population_size, 
        courses=courses, 
        class_times=class_times, 
        rooms=rooms, 
        instructors=instructors
    )
    scheduling_ga.evaluate_chromosomes()

    scheduling_ga.display_chromosome(scheduling_ga.population[0])

    return

if __name__ == "__main__":
    main()