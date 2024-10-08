
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
import matplotlib.pyplot as plt

City = namedtuple("City", ("name", "x", "y"))
firstSet = [
    City("Nazret", 10, 32),
    City("Gambela", 12, 29),
    City("Dessie", 14, 30),
    City("Bishoftu", 16, 34),
    City("Mendi", 18, 31),
    City("Shashemene", 20, 36),
    City("Harar", 22, 28),
    City("Asella", 24, 33),
    City("Negele Boran", 26, 34),
    City("Sodo", 28, 35),
    City("Woldiya", 30, 30),
    City("Kibre Mengist", 32, 31),
    City("Adama", 34, 37),
    City("Jijiga", 36, 29),
    City("Dembidolo", 38, 36),
    City("Gore", 40, 32),
    City("Bonga", 42, 30),
    City("Debre Birhan", 44, 31),
    City("Bati", 46, 33),
    City("Mekele", 48, 34),
]
cities2 = [i for i in range(len(firstSet))]

population_limit = 50

def get_distance(city1:int, city2:int, cities:List[City])->float:
    if cities[city1].name == cities[city2].name:
       return math.inf
    return round(math.sqrt((cities[city1].x - cities[city2].x)**2 + (cities[city1].y - cities[city2].y)**2),2)

def generate_population(cities: Genome, population_limit: int) -> Population:
    n = len(cities)
    
    # Validate that the population limit does not exceed the number of unique permutations
    if population_limit > math.factorial(n):
        raise ValueError("Population limit exceeds the number of unique permutations.")

    # Generate a list to hold the unique populations
    population = set()  # Using a set to avoid duplicates

    while len(population) < population_limit:
        # Generate a random array of unique indices
        random_indices = sample(range(n), n)
        # Create a new permutation based on the random indices
        new_permutation = [cities[i] for i in random_indices]
        population.add(tuple(new_permutation))  # Convert to tuple for set storage

    return [list(item) for item in population] 
def selection_pair(population: Population, fitness_func: FitnessFunc) -> Population:
    return choices(
        population=population, weights=[(1/fitness_func(gene)) for gene in population], k=2
    )
def calculate_total_distance(order: Genome,cities:List[City]) -> float:
    distance =  sum(get_distance(order[n], order[n + 1],cities=cities) for n in range(len(order) - 1))
    # Add the distance from the last city to the first
    distance += get_distance(order[-1], order[0],cities=cities)
    return distance

def fitness(order: Genome,cities:List[City]) -> float:
    """
    The fitness function just measures the total distance between the cities in the order.
    """
    if len(set(order)) != len(cities):
        return math.inf
    return calculate_total_distance(order,cities=cities)
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

def print_generation_info(generation: int, best_order: Genome, cities: List[City]):
    total_distance = calculate_total_distance(best_order, cities=cities)
    
    print(f"Generation: {generation}")
    print(f"Total Distance: {total_distance}")
    print("-" * 40)


def print_final_result(best_order: Genome, generation: int, cities:List[City]):
    total_distance = calculate_total_distance(best_order, cities=cities)
    
    print("\nFinal Solution:")
    print(f"Best Route:")
    best_route_str = " -> ".join([cities[i].name for i in best_order])
    print(f"Best Route: {best_route_str}")
    print(f"Total Distance: {total_distance}")
    print(f"Found in Generation: {generation}")
    print(f"Out of Possible Combinations: {math.factorial(len(best_order))}")
    print("=" * 40)

def visualize_route1(cities: List[City], best_route: List[int], generation: int, total_distance: float):
    """
    Visualize the cities and the best route for the Travelling Salesman Problem, with generation and distance info below the graph.
    Args:
        cities (List[City]): List of cities with coordinates.
        best_route (List[int]): List of indices representing the order of cities to visit.
        generation (int): The generation number in the genetic algorithm.
        total_distance (int): The total distance of the best route found.
    """
    # Extract the coordinates of the cities based on the best route
    x_coords = [cities[i].x for i in best_route]
    y_coords = [cities[i].y for i in best_route]

    # Close the loop by adding the first city at the end to return to the start
    x_coords.append(cities[best_route[0]].x)
    y_coords.append(cities[best_route[0]].y)

    # Create the plot
    plt.figure(figsize=(8, 6))
    
    # Plot the cities
    plt.scatter([city.x for city in cities], [city.y for city in cities], color="red")
    
    # Annotate the cities with names
    for i, city in enumerate(cities):
        plt.annotate(city.name, (city.x, city.y), textcoords="offset points", xytext=(0, 10), ha="center")
    
    # Plot the best route as a line connecting the cities
    plt.plot(x_coords, y_coords, color="blue", linestyle='-', marker='o', markersize=5)

    # Label the plot
    plt.title("Best Route for the Travelling Salesman Problem")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")

    # Add text with details about the final solution
    best_route_str = " -> ".join([cities[i].name for i in best_route])
    text = (
        f"Final Solution:\n"
        f"Best Route: {best_route_str}\n"
        f"Total Distance: {total_distance} km\n"
        f"Found in Generation: {generation}\n"
        f"Out of Possible Combinations: {math.factorial(len(best_route))}"
    )

    # Place the text below the graph using plt.figtext
    plt.figtext(0.1, 0.01, text, wrap=True, horizontalalignment='left', fontsize=10)

# Adjust the layout to make more space at the bottom for the text
    plt.subplots_adjust(bottom=0.2)

    # Show the plot
    plt.grid(True)
    plt.show()

# Run the genetic algorithm
def run_evolution(
    populate_func: PopulateFunc,
    fitness_func: FitnessFunc,
    selection_func: SelectionFunc = selection_pair,
    crossover_func: CrossoverFunc = single_point_crossover,
    mutation_func: MutationFunc = mutation,
    generation_limit: int = 100,
) -> Tuple[Population, int]:
    population = populate_func()
    for i in range(generation_limit):
        population = sort_population(population, fitness_func)
        next_generation = population[0:2]

        for _ in range(int(len(population) / 2) - 1):
            parents = selection_func(population, fitness_func)
            offspring_a, offspring_b = crossover_func(parents[0], parents[1])
            offspring_a = mutation_func(offspring_a)
            offspring_b = mutation_func(offspring_b)
            next_generation += [offspring_a, offspring_b]

        population = next_generation
        if i % 50 == 0:
            print_generation_info(i, population[0], firstSet)
    return population, i


print("---------- TRAVELLING SALESMAN PROBLEM ----------")
with timer():
    population, generation = run_evolution(
        partial(generate_population, cities=cities2, population_limit=population_limit),
        fitness_func=partial(fitness,cities=firstSet),
        selection_func=selection_pair,
        crossover_func=single_point_crossover,
        generation_limit=500,
    )
    print_final_result(population[0], generation, firstSet)

visualize_route1(firstSet, population[0],generation=generation, total_distance=calculate_total_distance(population[0], firstSet))

