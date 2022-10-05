from __future__ import annotations

from math import e
from statistics import mean, stdev
from random import choices
from typing import Callable

from .Chromosome import Chromosome
from .Gene import Gene

class GeneticAlgorithm:
    def __init__(
        self, 
        population_size: int,
        mutation_rate: float,
        possible_genes: dict[type, list[Gene]],
        chromosome_generator: Callable,
        chromosome_evaluator: Callable,
        chromosome_displayer: Callable,
        chromosome_mutator: Callable
    ) -> None:
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.possible_genes = possible_genes
        self.chromosome_generator = chromosome_generator
        self.chromosome_evaluator = chromosome_evaluator
        self.chromosome_displayer = chromosome_displayer
        self.chromosome_mutator = chromosome_mutator

        self.__population: list[Chromosome] = []
        self.__probablities: list[float] = []
        self.__fitnesses: list[float] = []

        self.__average_fitness: float = None
        self.__standard_deviation: float = None
        self.__fittest_chromosome: Chromosome = None

    @property
    def average_fitness(self) -> float:
        return self.__average_fitness
    
    @property
    def standard_deviation(self) -> float:
        return self.__standard_deviation

    @property
    def fitnesses(self) -> list[float]:
        return self.__fitnesses

    @property
    def fittest_chromosome(self) -> Chromosome:
        return self.__fittest_chromosome

    @property
    def population(self) -> list[Chromosome]:
        return self.__population

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # generate chromosomes

    def generate_chromosomes(self, count: int=1, **kwargs) -> None:
        return self.chromosome_generator(
            count=count, possible_genes=self.possible_genes, **kwargs
        )

    def initialize_population(self, **kwargs) -> None:
        self.__population = self.generate_chromosomes(**kwargs)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # evaluate chromosomes

    def evaluate_chromosomes(self) -> None:
        """
        Assign a fitness value to each chromosome in the population.
        """
        [self.chromosome_evaluator(c) for c in self.__population]

        # calculate fittest chromosome, mean, and standard deviation
        self.__fitnesses: list[float] = []
        self.__fittest_chromosome: Chromosome = Chromosome(genes=[])
        self.__fittest_chromosome.fitness = float('-inf')
        for chromosome in self.__population:
            self.__fitnesses.append(chromosome.fitness)
            self.__fittest_chromosome = max(
                self.__fittest_chromosome, chromosome, key=lambda c: c.fitness
            )

        # self.__fitnesses = [c.fitness for c in self.__population]
        # self.__fittest_chromosome = max(self.__population, key=lambda c: c.fitness)
        self.__average_fitness = mean(self.fitnesses)
        self.__standard_deviation = stdev(self.fitnesses)

    def calculate_probabilities(self) -> None:
        """
        Calculates a probability distribution of the chromosomes.
        """
        # Note, the order of the distribution aligns with the population.
        probablities: list[float] = [e ** c.fitness for c in self.__population]
        sum_probabilities: float = sum(probablities)
        self.probablities = [p / sum_probabilities for p in probablities]

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # create offspring

    def create_offspring(self) -> None:
        """
        Create a new population using the probability distribution to randomly
        select chromosomes, and then try to mutate them.
        """
        self.__population = [
            self.chromosome_mutator(
                chromosome=chromosome,
                mutation_rate=self.mutation_rate,
                possible_genes=self.possible_genes
            ) 
            for chromosome 
            in choices(
                population=self.__population,
                weights=self.probablities,
                k=self.population_size
            )
        ]

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # display choromosomes

    def display_chromosome(self, chromosome: Chromosome) -> None:
        self.chromosome_displayer(chromosome)