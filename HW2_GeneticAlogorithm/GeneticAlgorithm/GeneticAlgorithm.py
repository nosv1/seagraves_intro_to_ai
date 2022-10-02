from __future__ import annotations

from math import e
from statistics import mean, stdev
from random import random
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

        self.population: list[Chromosome] = []
        self.probablities: list[float] = []

        self.__average_fitness: float = None
        self.__standard_deviation: float = None

    @property
    def average_fitness(self) -> float:
        return self.__average_fitness
    
    @property
    def standard_deviation(self) -> float:
        return self.__standard_deviation

    def generate_chromosomes(self, count: int=1, **kwargs) -> None:
        return self.chromosome_generator(
            count=count, possible_genes=self.possible_genes, **kwargs
        )

    def initialize_population(self, **kwargs) -> None:
        self.population = self.generate_chromosomes(**kwargs)

    def evaluate_chromosomes(self) -> None:
        [self.chromosome_evaluator(c) for c in self.population]
        self.__average_fitness = mean([c.fitness for c in self.population])
        self.__standard_deviation = stdev([c.fitness for c in self.population])

    def display_chromosome(self, chromosome: Chromosome) -> None:
        self.chromosome_displayer(chromosome)

    def calculate_probabilities(self) -> None:
        """
        Calculates the probability of a chromosome being selected as a survivor
        using Softmax function.
        """
        self.probablities = [e ** c.fitness for c in self.population]
        sum_probabilities: float = sum(self.probablities)
        self.probablities = [p / sum_probabilities for p in self.probablities]
        self.probablities.sort()

    def create_offspring(self) -> None:
        """
        Creates the offspring using the Birther, then fill the population with
        the random chromosomes.
        """
        for i in range(self.population_size):
            r = random()
            for j, p in enumerate(self.probablities):
                min_p = 0 if not j else self.probablities[j-1]
                if min_p <= r < p:
                    self.population[i] = self.chromosome_mutator(
                        chromosome=self.population[j],
                        mutation_rate=self.mutation_rate,
                        possible_genes=self.possible_genes
                    )
                    break