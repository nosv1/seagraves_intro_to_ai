from __future__ import annotations

from functools import partial
from math import e
from multiprocessing import Pool
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

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # generate chromosomes

    def _chromosome_generator_wrapper(self, arg) -> Chromosome:
        args, kwargs = arg
        return self.chromosome_generator(*args, **kwargs)

    def generate_chromosomes(self, count: int=1, **kwargs) -> list[Chromosome]:
        # https://stackoverflow.com/questions/34031681/passing-kwargs-with-multiprocessing-pool-map
        # I'm still trying to munderstand what's going on here, but this works...

        # Because, we're trying to generalize the GeneticAlgorithm class, we need
        # to be able to pass in any number of arguments to the user defeined 
        # chromosome_generator. And given .map's callable can only take one 
        # argument, we have to create that argument.

        # The link provides an example where:
        #   arg = [(j, kwargs) for j in jobs]

        # but my arguments come from outside the class, and I just need to create
        # an arg iterable 'count' long, so I don't have any class defined args 
        # but I do have user defined kwargs... or something.
        # So, I think, if I did have args I wanted to pass in from within the
        # the class, they'd go in that empty tuple where 'j' is in the example.

        arg = [((), kwargs) for _ in range(count)]
        with Pool() as pool:
            chromosomes: list[Chromosome] = list(pool.imap_unordered(
                self._chromosome_generator_wrapper, arg
            ))
            return chromosomes

    def initialize_population(self, **kwargs) -> None:
        self.population = self.generate_chromosomes(**kwargs)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # evaluate chromosomes

    def evaluate_chromosomes(self) -> None:
        with Pool() as pool:
            self.population = list(pool.imap_unordered(
                self.chromosome_evaluator, self.population
            ))
        population_fitness: list[float] = [c.fitness for c in self.population]
        self.__average_fitness = mean(population_fitness)
        self.__standard_deviation = stdev(population_fitness)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # create offspring


    def calculate_probabilities(self) -> None:
        """
        Calculates the probability of a chromosome being selected for mutation.
        """
        self.probablities = [e ** c.fitness for c in self.population]
        sum_probabilities: float = sum(self.probablities)
        self.probablities = [p / sum_probabilities for p in self.probablities]
        self.probablities.sort()

    def _chromosome_mutator_wrapper(self, arg) -> Chromosome:
        args, kwargs = arg
        return self.chromosome_mutator(*args, **kwargs)

    def create_offspring(self) -> None:
        """
        Loops population and randomly chooses and chromosome to try and mutate.
        """
        arg = [
            ((), {
                'chromosome': chromosome,
                'mutation_rate': self.mutation_rate,
                'possible_genes': self.possible_genes
            })
            for chromosome
            in choices(
                population=self.population, 
                weights=self.probablities, 
                k=self.population_size
            )
        ]
        with Pool() as pool:
            self.population = list(pool.imap_unordered(
                self._chromosome_mutator_wrapper, arg
            ))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # display choromosomes

    def display_chromosome(self, chromosome: Chromosome) -> None:
        self.chromosome_displayer(chromosome)