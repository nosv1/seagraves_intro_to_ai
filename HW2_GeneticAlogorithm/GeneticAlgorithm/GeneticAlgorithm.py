from __future__ import annotations

from typing import Callable

from .Chromosome import Chromosome
from .Gene import Gene

class GeneticAlgorithm:
    def __init__(
        self, 
        population_size: int,
        mutation_rate: float,
        chromosome_generator: Callable,
        chromosome_evaluator: Callable,
        chromosome_displayer: Callable,
    ) -> None:
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.gene_generator = chromosome_generator
        self.chromosome_evaluator = chromosome_evaluator
        self.chromosome_displayer = chromosome_displayer

        self.population: list[Chromosome] = []

    def generate_chromosomes(self, count: int=1, **kwargs) -> None:
        return self.gene_generator(count=count, **kwargs)

    def initialize_population(self, **kwargs) -> None:
        self.population = self.generate_chromosomes(**kwargs)

    def evaluate_chromosomes(self) -> None:
        [self.chromosome_evaluator(c) for c in self.population]

    def display_chromosome(self, chromosome: Chromosome) -> None:
        self.chromosome_displayer(chromosome)