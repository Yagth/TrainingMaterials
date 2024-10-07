# Gnome representation
# Generating population
# crossover
# Mutation
# Population Fitness
# selection pair
# Run genetic algorithm

import time
from functools import partial
from typing import Callable, List, Optional, Tuple
from collections import namedtuple
from random import choices, randint, random, randrange

from contextlib import contextmanager

weight_limit = 3000

Genome = List[int]
Population = List[Genome]
PopulateFunc = Callable[[], Population]
FitnessFunc = Callable[[Genome], int]
SelectionFunc = Callable[[Population, FitnessFunc], Population]
CrossoverFunc = Callable[[Genome, Genome], Tuple[Genome, Genome]]
MutationFunc = Callable[[Genome], Genome]
PrinterFunc = Callable[[Population, int, FitnessFunc], None]

Thing = namedtuple("Thing", ("name", "value", "weight"))

first_example = [
    Thing("Laptop", 500, 2200),
    Thing("Headphones", 150, 160),
    Thing("Coffee Mug", 60, 350),
    Thing("Notepad", 40, 333),
    Thing("Water Bottle", 30, 192),
]

second_example = [
    Thing("Mints", 5, 25),
    Thing("Socks", 10, 38),
    Thing("Tissues", 15, 80),
    Thing("Phone", 500, 200),
    Thing("Baseball Cap", 100, 70),
] + first_example


@contextmanager
def timer():
    start = time.time()
    yield
    end = time.time()
    print(f"Elapsed Time: {(end - start)}s")


def generate_things(num: int) -> List[Thing]:
    return [Thing(f"thing{i}", i, i) for i in range(1, num + 1)]


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


def fitness(genome: Genome, things: List[Thing], weight_limit: int) -> int:
    if len(genome) != len(things):
        raise ValueError("genome and things should be equal in length")

    weight = 0
    value = 0

    for i, thing in enumerate(things):
        if genome[i] == 1:
            weight += thing.weight
            value += thing.value

            if weight > weight_limit:
                return 0

    return value


def population_fitness(population: Population, fitness_func: FitnessFunc) -> int:
    return sum([fitness_func(gene) for gene in population])


def selection_pair(population: Population, fitness_func: FitnessFunc) -> Population:
    return choices(
        population=population, weights=[fitness_func(gene) for gene in population], k=2
    )


def sort_population(population: Population, fitness_func: FitnessFunc) -> Population:
    return sorted(population, key=fitness_func, reverse=True)


def genome_to_string(genome: Genome) -> str:
    return "".join(map(str, genome))


def print_stats(population: Population, generation_id: int, fitness_func: FitnessFunc):
    print("Generation %02d" % generation_id)
    print("=" * 50)
    print(
        "Population: [%s]" % ", ".join([genome_to_string(gene) for gene in population])
    )
    print(
        "Avg. Fitness: %f"
        % (population_fitness(population, fitness_func) / len(population))
    )
    sorted_population = sort_population(population, fitness_func)
    print(
        "Best: %s (%f)"
        % (genome_to_string(sorted_population[0]), fitness_func(sorted_population[0])),
    )
    print(
        "worst: %s (%f)"
        % (
            genome_to_string(sorted_population[-1]),
            fitness_func(sorted_population[-1]),
        ),
    )

    print("")


def print_things_stats(things: List[Thing]):
    print(f"Things: {to_string(things)}")
    print(f"Value {value(things)}")
    print(f"Weight: {weight(things)}")


def from_genome(genome: Genome, things: List[Thing]) -> List[Thing]:
    result = []
    for i, thing in enumerate(things):
        if genome[i] == 1:
            result.append(thing)

    return result


def to_string(things: List[Thing]) -> str:
    return f"[{', '.join([t.name for t in things])}]"


def weight(things: List[Thing]) -> int:
    return sum([t.weight for t in things])


def value(things: List[Thing]) -> int:
    return sum([t.value for t in things])


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


print("Weight Limit: %dkg" % weight_limit)
print("")
print("GENETIC ALGORITHM")
print("----------")

with timer():
    things = generate_things(32)
    population, generation = run_evolution(
        partial(generate_population, size=10, genome_length=len(second_example)),
        partial(fitness, things=second_example, weight_limit=weight_limit),
        fitness_limit=1310,
        generation_limit=100,
    )

sack = from_genome(population[0], second_example)
print_things_stats(sack)
