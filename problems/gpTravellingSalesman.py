
import math
import sys
import os
# Add the parent directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from genetic_algorithm.types import  Genome, Population, FitnessFunc, SelectionFunc, PopulateFunc, CrossoverFunc, MutationFunc
from genetic_algorithm.utils import timer
from genetic_algorithm.evolution import generate_population
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple
from collections import namedtuple
from random import choices, randint, random, randrange,sample
from itertools import permutations

Distances = Dict[Tuple[int,int],int]

# City = namedtuple("City",("name", "x", "y"))
# distances = {
#     (1, 2): 10,
#     (1, 3): 15,
#     (1, 4): 20,
#     (2, 3): 35,
#     (2, 4): 25,
#     (3, 4): 30
# }
# cities = [1,2,3,4]
cities2 = [1,2,3,4,5,6,7,8,9,10]
distances2 = {
    (1, 2): 100,  # Addis Ababa to Dire Dawa
    (1, 3): 25,  # Addis Ababa to Hawassa
    (1, 4): 40,  # Addis Ababa to Gondar
    (1, 5): 55,  # Addis Ababa to Bahir Dar
    (1, 6): 80,  # Addis Ababa to Jimma
    (1, 7): 65,  # Addis Ababa to Axum
    (1, 8): 35,  # Addis Ababa to Debre Markos
    (1, 9): 70,  # Addis Ababa to Arba Minch
    (1, 10): 90,  # Addis Ababa to Mekelle
    (2, 3): 80,  # Dire Dawa to Hawassa
    (2, 4): 120,  # Dire Dawa to Gondar
    (2, 5): 135,  # Dire Dawa to Bahir Dar
    (2, 6): 100,  # Dire Dawa to Jimma
    (2, 7): 145,  # Dire Dawa to Axum
    (2, 8): 110,  # Dire Dawa to Debre Markos
    (2, 9): 150,  # Dire Dawa to Arba Minch
    (2, 10): 130,  # Dire Dawa to Mekelle
    (3, 4): 60,  # Hawassa to Gondar
    (3, 5): 75,  # Hawassa to Bahir Dar
    (3, 6): 40,  # Hawassa to Jimma
    (3, 7): 105,  # Hawassa to Axum
    (3, 8): 30,  # Hawassa to Debre Markos
    (3, 9): 80,  # Hawassa to Arba Minch
    (3, 10): 110,  # Hawassa to Mekelle
    (4, 5): 15,  # Gondar to Bahir Dar
    (4, 6): 100,  # Gondar to Jimma
    (4, 7): 120,  # Gondar to Axum
    (4, 8): 80,  # Gondar to Debre Markos
    (4, 9): 150,  # Gondar to Arba Minch
    (4, 10): 130,  # Gondar to Mekelle
    (5, 6): 80,  # Bahir Dar to Jimma
    (5, 7): 100,  # Bahir Dar to Axum
    (5, 8): 60,  # Bahir Dar to Debre Markos
    (5, 9): 130,  # Bahir Dar to Arba Minch
    (5, 10): 110,  # Bahir Dar to Mekelle
    (6, 7): 120,  # Jimma to Axum
    (6, 8): 70,  # Jimma to Debre Markos
    (6, 9): 80,  # Jimma to Arba Minch
    (6, 10): 100,  # Jimma to Mekelle
    (7, 8): 100,  # Axum to Debre Markos
    (7, 9): 180,  # Axum to Arba Minch
    (7, 10): 30,  # Axum to Mekelle
    (8, 9): 100,  # Debre Markos to Arba Minch
    (8, 10): 80,  # Debre Markos to Mekelle
    (9, 10): 120  # Arba Minch to Mekelle
}
population_limit = 50

def get_distance(city1:int, city2:int, distances:Distances)->int:
    if (city1, city2) in distances:
        return distances[(city1, city2)]
    elif (city2, city1) in distances:
        return distances[(city2, city1)]
    else:
        return math.inf
def generate_gnome(length: int) -> Genome:
    return choices([0, 1], k=length)
def generate_population(cities: Genome, population_limit: int) -> Population:
    all_permutations = list(permutations(cities))
    
    # If the number of desired permutations exceeds the total number of permutations
    if population_limit > len(all_permutations):
        return all_permutations
    # Randomly select the desired number of permutations
    return sample(all_permutations, population_limit)
def selection_pair(population: Population, fitness_func: FitnessFunc) -> Population:
    return choices(
        population=population, weights=[(1/fitness_func(gene)) for gene in population], k=2
    )
def calculate_total_distance(order: Genome,distances:Distances) -> int:
    distance =  sum(get_distance(order[n], order[n + 1],distances=distances) for n in range(len(order) - 1))
    # Add the distance from the last city to the first
    distance += get_distance(order[-1], order[0],distances=distances)
    return distance

def fitness(order: Genome,distances:Distances,cities:Genome) -> int:
    if len(set(order)) != len(cities):
        return math.inf
    return calculate_total_distance(order,distances)
def single_point_crossover(a: Genome, b: Genome) -> Tuple[Genome, Genome]:
    a=list(a)
    b=list(b)
    if len(a) != len(b):
        ValueError("Genomes a and b must be the same length")

    length = len(a)
    if length < 2:
        return a, b

    p = randint(1, length - 1)
    return a[:p] + b[p:], b[:p] + a[p:]

def sort_population(population: Population, fitness_func: FitnessFunc) -> Population:
    return sorted(population, key=fitness_func, reverse=False)

print("TRAVELLING SALESMAN PROBLEM")
print("----------")

def printer(order:Genome,distances:Distances):
    distance = calculate_total_distance(order,distances=distances)
    print(f"Order: {order} Distance: {distance}")
def mutation(genome: Genome, num: int = 1, probability: float = 0.5) -> Genome:
    genome = list(genome)
    for _ in range(num):
        index1 = randint(0,int(len(genome)/2)-1)
        index2=randint(int(len(genome)/2),len(genome)-1)
        if random() > probability:
            val1= genome[index1]
            val2 = genome[index2]
            genome[index1] =val2
            genome[index2] =val1
    return genome
def run_evolution(
    populate_func: PopulateFunc,
    fitness_func: FitnessFunc,
    # fitness_limit: int,
    selection_func: SelectionFunc = selection_pair,
    crossover_func: CrossoverFunc = single_point_crossover,
    mutation_func: MutationFunc = mutation,
    generation_limit: int = 100,
    # printer: Optional[PrinterFunc] = None,
) -> Tuple[Population, int]:
    population = populate_func()
    for n in population:
        distance = fitness_func(n)
        print(f"Order123: {n} Distance: {distance}")
    for i in range(generation_limit):
        population = sort_population(population, fitness_func)

        # if printer is not None:
        #     printer(population, i, fitness_func)
        # if fitness_func(population[0]) >= fitness_limit:
        #     break

        next_generation = population[0:2]

        for _ in range(int(len(population) / 2) - 1):
            parents = selection_func(population, fitness_func)
            offspring_a, offspring_b = crossover_func(parents[0], parents[1])
            offspring_a = mutation_func(offspring_a)
            offspring_b = mutation_func(offspring_b)
            next_generation += [offspring_a, offspring_b]

        population = next_generation

    return population, i
with timer():
    population, generation = run_evolution(
        partial(generate_population, cities=cities2, population_limit=population_limit),
        fitness_func=partial(fitness, distances=distances2,cities=cities2),
        selection_func=selection_pair,
        crossover_func=single_point_crossover,
        generation_limit=1000,
    )

print(f"{printer(population[0],distances2)},Generation no :{generation}, Possible Combination: {math.factorial(len(cities2))}")