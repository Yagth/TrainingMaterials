import random
from typing import List, Tuple, Callable
import sys
import os
# Add the parent directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from genetic_algorithm.types import  Genome, Population, FitnessFunc
# Define type aliases


POPULATION_SIZE = 500
MUTATION_PROBABILITY = 0.1

# Generate a random genome (chromosome)
def generate_random_genome(size: int) -> Genome:
    return [random.randint(0, size - 1) for _ in range(size)]

# Calculate the fitness of a genome
def fitness(genome: Genome, max_fitness: int) -> int:
    n = len(genome)
    horizontal_collisions = sum([genome.count(queen) - 1 for queen in genome]) // 2

    left_diagonal = [0] * (2 * n - 1)
    right_diagonal = [0] * (2 * n - 1)
    for i in range(n):
        left_diagonal[i + genome[i] - 1] += 1
        right_diagonal[n - i + genome[i] - 2] += 1

    diagonal_collisions = sum([count - 1 for count in left_diagonal if count > 1]) + \
                          sum([count - 1 for count in right_diagonal if count > 1])

    return int(max_fitness - (horizontal_collisions + diagonal_collisions))

# Perform crossover between two genomes
def single_point_crossover(genome_a: Genome, genome_b: Genome) -> Genome:
    n = len(genome_a)
    return [genome_a[i] if random.random() < 0.5 else genome_b[i] for i in range(n)]

# Mutate a genome
def mutation(genome: Genome) -> Genome:
    n = len(genome)
    genome[random.randint(0, n - 1)] = random.randint(0, n - 1)
    return genome

# Calculate the probability for selection based on fitness
def calculate_probability(genome: Genome, max_fitness: int) -> float:
    return fitness(genome, max_fitness) / max_fitness

# Select a random genome based on fitness probabilities (roulette wheel selection)
def selection_pair(population: Population, fitness_func: FitnessFunc, max_fitness: int) -> Tuple[Genome, Genome]:
    probabilities = [fitness_func(genome, max_fitness) / max_fitness for genome in population]
    return random.choices(population, weights=probabilities, k=2)

# Generate the next generation of genomes
def evolve_population(population: Population, fitness_func: FitnessFunc, max_fitness: int) -> Population:
    next_generation = []
    probabilities = [fitness(genome, max_fitness) / max_fitness for genome in population]

    # Preserve elitism (best and worst individuals)
    sorted_population = sorted(population, key=lambda chromo: fitness_func(chromo, max_fitness), reverse=True)
    next_generation.append(sorted_population[0])
    next_generation.append(sorted_population[-1])

    # Generate new individuals via crossover and mutation
    for _ in range(len(population) - 2):
        parent_a, parent_b = selection_pair(population, fitness_func, max_fitness)
        offspring = single_point_crossover(parent_a, parent_b)
        if random.random() < MUTATION_PROBABILITY:
            offspring = mutation(offspring)
        next_generation.append(offspring)

    return next_generation

# Print genome and fitness
def print_genome(genome: Genome, max_fitness: int) -> None:
    print(f"Genome = {genome}, Fitness = {fitness(genome, max_fitness)}")

# Print chessboard for given genome
def print_board(genome: Genome) -> None:
    n = len(genome)
    board = [["x"] * n for _ in range(n)]
    for i in range(n):
        board[genome[i]][i] = "Q"
    
    for row in board:
        print(" ".join(row))
    print()

# Run the N-Queens genetic algorithm
def run_genetic_algorithm(n_queens: int) -> None:
    max_fitness = (n_queens * (n_queens - 1)) // 2
    population = [generate_random_genome(n_queens) for _ in range(POPULATION_SIZE)]
    generation = 1

    while max_fitness not in [fitness(gene, max_fitness) for gene in population] and generation < 100:
        population = evolve_population(population, fitness, max_fitness)

        if generation % 10 == 0:
            print(f"=== Generation {generation} ===")
            best_fitness = max([fitness(chromo, max_fitness) for chromo in population])
            print(f"Maximum Fitness = {best_fitness}")

        generation += 1

    fitness_scores = [fitness(chromo, max_fitness) for chromo in population]
    best_genome = population[fitness_scores.index(max(fitness_scores))]

    if max_fitness in fitness_scores:
        print(f"\nSolved in Generation {generation - 1}!")
    else:
        print(f"\nNo solution found after {generation - 1} generations.")

    print_genome(best_genome, max_fitness)
    print_board(best_genome)


while True:
    n_queens = int(input("Enter the number of queens (0 to exit): "))
    if n_queens == 0:
        break
    run_genetic_algorithm(n_queens)
