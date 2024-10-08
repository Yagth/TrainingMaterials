import random
from time import time
from typing import List, Tuple
import sys
import os
# Add the parent directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from genetic_algorithm.types import  Genome, Population, FitnessFunc


# Data for the scheduling problem
T = 4  # Number of jobs
operations_per_job = [2, 3, 4, 2]  # Number of operations per job
ma = 6  # Number of machines
machine_operations = [
    [1, 2, 3, 4, 5], [1, 3, 4, 6], [1, 3, 2], [1, 2, 5], [1, 2, 3, 4], [1, 2, 5],
    [1, 2, 3, 6], [1, 3, 5], [1, 5, 6], [1, 6], [2, 3, 4]
]
processing_times = [
    [3, 4, 3, 4, 4, 1000], [5, 1000, 5, 4, 1000, 4], [3, 4, 6, 1000, 1000, 1000],
    [2, 4, 1000, 1000, 4, 1000], [1, 3, 3, 2, 1000, 1000], [1, 3, 1000, 1000, 2, 1000],
    [2, 2, 2, 1000, 1000, 2], [1, 1000, 1, 1000, 2, 1000], [4, 1000, 1000, 1000, 3, 3],
    [3, 1000, 1000, 1000, 1000, 4], [1000, 5, 3, 4, 1000, 1000]
]

# Validate data correctness
def validate_data(data: List) -> None:
    total_operations = sum(data[1])
    if len(data[1]) != data[0]:
        print("Error: Mismatch between jobs and operations list.")
        exit()
    if len(data[3]) != total_operations:
        print("Error: Mismatch between operations and machine assignment list.")
        exit()
    if len(data[4]) != total_operations:
        print("Error: Mismatch between operations and processing time list.")
        exit()

data = [T, operations_per_job, ma, machine_operations, processing_times]
validate_data(data)

# Maximum processing time
def max_processing_time(data: List) -> int:
    return max(max(p for p in job if p != 1000) for job in data[4])

# Generate a random genome
def generate_random_genome(data: List) -> Genome:
    """
    The genome generation function creates a random valid solution (genome) for the job shop scheduling problem.

    1. **Objective**: Generate an initial population of random schedules (genomes) for the genetic algorithm.
    2. **Representation**: The genome is a list where even indices represent start times, and odd indices represent machines.

    Args:
        data (List): The problem data, including job details, available machines, and processing times.

    Returns:
        Genome: A list representing the genome, structured as [start_time_1, machine_1, start_time_2, machine_2, ...].
    """

    individual = []  # Represents the genome (solution)
    start_times = [0] * data[2]  # Tracks the start times for each machine (initially 0)
    jobs = data[0]  # Number of jobs in the problem

    # Loop through each job and generate a random schedule for its operations
    for i in range(jobs):
        # For each operation in the job
        for j in range(data[1][i]):
            operation = random.randint(0, len(data[3]) - 1)  # Randomly select an operation index
            machine_idx = random.choice(data[3][operation]) - 1  # Randomly select a machine for the operation
            start_time = start_times[machine_idx]  # Get the current start time for the machine
            individual.append(start_time)  # Add start time to the genome
            individual.append(machine_idx + 1)  # Add machine to the genome

            # Update the start time for the machine based on the operation's processing time
            start_times[machine_idx] += data[4][operation][machine_idx]
    
    return individual  # Return the generated genome

# Mutate a genome
def mutate_genome(genome: Genome, data: List) -> None:
    mutate_index1 = random.randrange(len(genome))
    mutate_index2 = random.randrange(len(genome))

    if (mutate_index1 % 2 == mutate_index2 % 2):
        genome[mutate_index1], genome[mutate_index2] = genome[mutate_index2], genome[mutate_index1]
    elif mutate_index1 % 2 == 0:
        genome[mutate_index1], genome[random.randrange(0, len(genome), 2)] = genome[random.randrange(0, len(genome), 2)], genome[mutate_index1]
        genome[mutate_index2] = random.randint(1, data[2])
    else:
        genome[mutate_index2], genome[random.randrange(0, len(genome), 2)] = genome[random.randrange(0, len(genome), 2)], genome[mutate_index2]
        genome[mutate_index1] = random.randint(1, data[2])

# Fitness function for evaluating a genome
def fitness(genome: Genome, data: List) -> int:
    processing_times = data[4]
    fouls = 0
    fitness_value = 0

    for operation_idx, (start_time, machine) in enumerate(zip(genome[::2], genome[1::2])):
        fitness_value = max(fitness_value, start_time + processing_times[operation_idx][machine - 1])

    for job_start, job_len in zip(range(0, len(genome), 2), operations_per_job):
        for operation_idx in range(job_start, job_start + (job_len - 1) * 2, 2):
            if genome[operation_idx] >= genome[operation_idx + 2]:
                fouls += 4

    for operation_idx in range(0, len(genome), 2):
        if genome[operation_idx + 1] not in data[3][operation_idx // 2]:
            fouls += 2

    fitness_value += fouls * 1000
    return fitness_value

# --- Helper Functions for Visualization and Explanation ---

def describe_problem(data: List) -> None:
    """
    Provide a textual description of the job shop scheduling problem.
    """
    print("=== Job Shop Scheduling Problem ===")
    print(f"Number of jobs: {data[0]}")
    print(f"Number of machines: {data[2]}")
    print("Operations for each job:")
    for i, job_operations in enumerate(data[1], start=1):
        print(f"  Job {i}: {job_operations} operations")
    print("\nMachine availability and processing times per operation:")

    for i, (machines, processing_times) in enumerate(zip(data[3], data[4])):
        machine_times = ", ".join(f"Machine {m}: {p} time" for m, p in zip(machines, processing_times) if p != 1000)
        print(f"  Operation {i + 1}: {machine_times}")

def explain_solution(genome: Genome, data: List) -> None:
    """
    Convert the genome into a human-readable schedule and explain the best solution.
    """
    print("\n=== Solution Explanation ===")
    for i in range(0, len(genome), 2):
        start_time = genome[i]
        machine = genome[i + 1]
        operation = i // 2 + 1
        print(f"Operation {operation} assigned to Machine {machine} starting at time {start_time}")

def explain_fitness_score(fitness_value: int) -> None:
    """
    Explain the meaoperations_per_jobng of the fitness score.
    """
    print("\n=== Fitness Score ===")
    print(f"The fitness score is: {fitness_value}")
    print("Lower scores are better. Penalties are applied for:")
    print("  - Operations scheduled out of order")
    print("  - Operations assigned to unavailable machines")
    print("  - Overlapping operations on the same machine")

# --- Genetic Algorithm for Job Shop Scheduling ---
def genetic_algorithm_scheduling(
    data: List, population_size: int = 100, generations: int = 500
) -> Tuple[Genome, int]:
    population = [generate_random_genome(data) for _ in range(population_size)]
    best_fitness_per_generation = []

    for generation in range(generations):
        population.sort(key=lambda individual: fitness(individual, data))
        best_fitness_per_generation.append(fitness(population[0], data))
        next_generation = population[:2]  # Elitism: retain best individuals

        for _ in range(int(population_size / 2) - 1):
            parent_a, parent_b = random.choices(population[:50], k=2)  # Select from top 50
            offspring = generate_random_genome(data)  # Recombine (could add crossover)
            mutate_genome(offspring, data)
            next_generation += [offspring]

        population = next_generation

    return population[0], best_fitness_per_generation[-1]

# Run the genetic algorithm
def run() -> None:
    describe_problem(data)
    result, fitness_value = genetic_algorithm_scheduling(data)
    explain_solution(result, data)
    explain_fitness_score(fitness_value)

if __name__ == "__main__":
    run()
