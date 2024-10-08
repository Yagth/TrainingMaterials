# problems/knapsack.py
from contextlib import contextmanager
from functools import partial
import sys
import os
import time
# Add the parent directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from genetic_algorithm.types import  Genome
from genetic_algorithm.utils import timer
from genetic_algorithm.evolution import run_evolution, generate_population
# from genetic_algorithm.types import Genome, Population, FitnessFunc
from random import choices
from typing import Callable, List, Optional, Tuple
from collections import namedtuple

Thing = namedtuple("Thing", ("name", "value", "weight"))

weight_limit = 3000

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

# @contextmanager
# def timer():
#     start = time.time()
#     yield
#     end = time.time()
#     print(f"Elapsed Time: {(end - start)}s")

def generate_things(num: int) -> List[Thing]:
    return [Thing(f"thing{i}", i, i) for i in range(1, num + 1)]

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

def print_things_stats(things: List[Thing]):
    print(f"Things: {to_string(things)}")
    print(f"Value {value(things)}")
    print(f"Weight: {weight(things)}")

# Running the evolution
print("Weight Limit: %dkg" % weight_limit)
print("")
print("GENETIC ALGORITHM")
print("----------")

with timer():
    things = generate_things(32)
    # print("timer started")
    population, generation = run_evolution(
        partial(generate_population, size=10, genome_length=len(second_example)),
        partial(fitness, things=second_example, weight_limit=weight_limit),
        fitness_limit=1310,
        generation_limit=100,
    )

sack = from_genome(population[0], second_example)
print_things_stats(sack)
