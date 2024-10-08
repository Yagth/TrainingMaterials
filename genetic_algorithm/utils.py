# genetic_algorithm/utils.py
import time
from contextlib import contextmanager
from genetic_algorithm.types import Genome, Population, FitnessFunc

@contextmanager
def timer():
    start = time.time()
    yield
    end = time.time()
    print(f"Elapsed Time: {(end - start)}s")

def sort_population(population: Population, fitness_func: FitnessFunc) -> Population:
    return sorted(population, key=fitness_func, reverse=True)

def population_fitness(population: Population, fitness_func: FitnessFunc) -> int:
    return sum([fitness_func(gene) for gene in population])

def genome_to_string(genome: Genome) -> str:
    return "".join(map(str, genome))
