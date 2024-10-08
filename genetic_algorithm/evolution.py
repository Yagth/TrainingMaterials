    # genetic_algorithm/evolution.py
from random import choices, randint, random, randrange
from genetic_algorithm.types import Genome, Population, FitnessFunc, PopulateFunc, SelectionFunc, CrossoverFunc, MutationFunc, PrinterFunc
from genetic_algorithm.utils import sort_population, population_fitness
from typing import Callable, List, Optional, Tuple

def generate_gnome(length: int) -> Genome:
    return choices([0, 1], k=length)

def generate_population(size: int, genome_length: int) -> Population:
    return [generate_gnome(genome_length) for _ in range(size)]

def single_point_crossover(a: Genome, b: Genome) -> Tuple[Genome, Genome]:
    if len(a) != len(b):
        raise ValueError("Genomes a and b must be the same length")

    length = len(a)
    if length < 2:
        return a, b

    p = randint(1, length - 1)
    return a[:p] + b[p:], b[:p] + a[p:]

def mutation(genome: Genome, num: int = 1, probability: float = 0.5) -> Genome:
    for _ in range(num):
        index = randrange(len(genome))
        genome[index] = (
            genome[index] if random() > probability else abs(genome[index] - 1)
        )
    return genome

def selection_pair(population: Population, fitness_func: FitnessFunc) -> Population:
    return choices(
        population=population, weights=[fitness_func(gene) for gene in population], k=2
    )

def run_evolution(
    populate_func: PopulateFunc,
    fitness_func: FitnessFunc,
    fitness_limit: int,
    selection_func: SelectionFunc = selection_pair,
    crossover_func: CrossoverFunc = single_point_crossover,
    mutation_func: MutationFunc = mutation,
    generation_limit: int = 100,
    printer: Optional[PrinterFunc] = None,
) -> Tuple[Population, int]:
    population = populate_func()

    for i in range(generation_limit):
        population = sort_population(population, fitness_func)

        if printer is not None:
            printer(population, i, fitness_func)

        if fitness_func(population[0]) >= fitness_limit:
            break

        next_generation = population[0:2]

        for _ in range(int(len(population) / 2) - 1):
            parents = selection_func(population, fitness_func)
            offspring_a, offspring_b = crossover_func(parents[0], parents[1])
            offspring_a = mutation_func(offspring_a)
            offspring_b = mutation_func(offspring_b)
            next_generation += [offspring_a, offspring_b]

        population = next_generation
    
    return population, i
