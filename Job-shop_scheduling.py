import matplotlib.pyplot as plt
from time import time
from pyeasyga import pyeasyga  
import random

# Input data for the scheduling problem
T = 4  # number of jobs
ni = [2, 3, 4, 2]  # number of operations per job
ma = 6  # number of machines
Mij = [
    [1, 2, 3, 4, 5], [1, 3, 4, 6], [1, 3, 2], [1, 2, 5], [1, 2, 3, 4], [1, 2, 5], [1, 2, 3, 6],
    [1, 3, 5], [1, 5, 6], [1, 6], [2, 3, 4]
]
pjk = [
    [3, 4, 3, 4, 4, 1000], [5, 1000, 5, 4, 1000, 4], [3, 4, 6, 1000, 1000, 1000], [2, 4, 1000, 1000, 4, 1000],
    [1, 3, 3, 2, 1000, 1000], [1, 3, 1000, 1000, 2, 1000], [2, 2, 2, 1000, 1000, 2], [1, 1000, 1, 1000, 2, 1000],
    [4, 1000, 1000, 1000, 3, 3], [3, 1000, 1000, 1000, 1000, 4], [1000, 5, 3, 4, 1000, 1000]
]


def is_data_ok(data):
    """
    Validates the input data for the Job Shop Scheduling Problem.

    The function checks the following conditions:
    1. The length of the 'ni' list (number of operations for each job) should match the number of jobs (data[0]).
    2. The length of the 'Mij' list (available machines for each operation) should match the total number of operations (sum of elements in 'ni').
    3. The length of the 'pjk' list (processing times for each operation) should also match the total number of operations.

    If any of these conditions are not met, the function will print an error message and terminate the program.

    Parameters:
    data (list): A list containing the following elements:
        - data[0] (int): Number of jobs.
        - data[1] (list of int): A list where each element indicates the number of operations for each job.
        - data[2] (int): Total number of machines (not used in this function but may be part of the input structure).
        - data[3] (list): A list of lists, where each sub-list indicates the machines available for each operation.
        - data[4] (list): A list of processing times for each operation.

    Returns:
    None: The function will exit if the data is invalid.
    """

    sum_ni = sum(data[1]) 
    if len(data[1]) != data[0]:
        print("Data invalid. Please check the length of ni list")
        exit()
    elif len(data[3]) != sum_ni:
        print("Data invalid. Please check the length of Mij list")
        exit()
    elif len(data[4]) != sum_ni:
        print("Data invalid. Please check the length of pjk list")
        exit()


data = [T, ni, ma, Mij, pjk]
is_data_ok(data)

def max_processing_time(data):
    """
    Calculates the maximum processing time of all operations in the job shop scheduling problem.

    The function iterates through the processing times of each operation and returns the highest 
    processing time found, excluding any placeholder values (in this case, 1000).

    Parameters:
    data (list): A list containing the following elements:
        - data[4] (list of lists): A list of processing times for each operation. 
          Each sub-list represents the processing times for the operations assigned to a specific job.

    Returns:
    int: The maximum processing time of all valid operations, excluding placeholder values.
    """
    
    pjk = data[4]  
    return max(max(p for p in job if p != 1000) for job in pjk)

def create_individual(data):
    """
    Creates a random individual (solution) for the job shop scheduling problem.

    This function generates a list representing the start times and assigned machines for each operation 
    of the jobs based on the input data. The individual is structured such that each job's operations 
    are assigned to specific machines with their corresponding start times.

    Parameters:
    data (list): A list containing the following elements:
        - data[0] (int): The total number of jobs.
        - data[1] (list of int): A list where each element represents the number of operations per job.
        - data[2] (int): The total number of machines.
        - data[3] (list of lists): A list where each sub-list contains the feasible machines for each operation.
        - data[4] (list of lists): A list where each sub-list contains the processing times for each operation.

    Returns:
    list: A list representing an individual, where each operation is assigned a start time and a machine.
           The structure is [start_time_1, machine_1, start_time_2, machine_2, ...].
    """

    individual = []  
    start_times = [0] * data[2]  
    jobs = data[0]  

    list_to = [2, 1, 2, 0, 1, 2, 0, 1, 1, 0]
    random_number = random.randint(0, len(list_to) - 1)  
    reference = list_to[random_number]  

    a = 0 if reference != 2 else len(data[3]) - 1
    direction = 1 if reference == 1 else -1

    for i in range(jobs):
        for j in range(data[1][i]):
            position_X = random.randint(0, len(data[3][a]) - 1)
            X = data[3][a][position_X]  
            S = start_times[X - 1]  
            
            individual.append(S)
            individual.append(X)
            
            start_times[X - 1] += data[4][a][X - 1]
            a += direction  

    return individual  


def mutate(individual):
    """
    Mutates a given individual (solution) for the job shop scheduling problem.

    This function introduces random changes to the individual's genes to maintain genetic diversity
    within the population. The mutation process involves swapping values or altering machine assignments
    based on specific conditions.

    Parameters:
    individual (list): A list representing an individual, structured as [start_time_1, machine_1, start_time_2, machine_2, ...].

    Returns:
    None: The mutation is applied in-place, modifying the original individual.
    """

    mutate_index1 = random.randrange(len(individual))
    mutate_index2 = random.randrange(len(individual))
    
    max_time = max_processing_time(data)

    if (mutate_index1 % 2 == mutate_index2 % 2):
        individual[mutate_index1], individual[mutate_index2] = individual[mutate_index2], individual[mutate_index1]
    elif mutate_index1 % 2 == 0:
        individual[mutate_index1], individual[random.randrange(0, len(individual), 2)] = individual[random.randrange(0, len(individual), 2)], individual[mutate_index1]
        individual[mutate_index2] = random.randint(1, data[2])  
    else:
        individual[mutate_index2], individual[random.randrange(0, len(individual), 2)] = individual[random.randrange(0, len(individual), 2)], individual[mutate_index2]
        individual[mutate_index1] = random.randint(1, data[2])  


def is_feasible_machine(operation, machine, data):
    """
    Checks if the specified machine can perform the given operation.

    Parameters:
    operation (int): Index of the operation; machine (int): Machine number; data (list): Contains machine assignment info.

    Returns:
    bool: True if the machine is feasible for the operation, False otherwise.
    """

    return machine in data[3][operation]

def operations_in_machine(machine, individual):
    """
    Retrieves indices of operations assigned to the specified machine.

    Parameters:
    machine (int): The machine number; individual (list): The individual representation containing operations.

    Returns:
    list: A list of operation indices assigned to the specified machine.
    """

    return [int(i / 2) for i in range(0, len(individual), 2) if individual[i + 1] == machine]

def fitness(individual, data):
    """
    Calculates the fitness value of an individual in a job shop scheduling problem.

    Parameters:
    individual (list or Individual): Represents the schedule of operations and their assigned machines.
    data (tuple): Contains information about job processing times, machine feasibility, etc.

    Returns:
    int: The computed fitness value, incorporating penalties for constraint violations.
    """

    if isinstance(individual, list):
        genes = individual  
    else:
        genes = individual.genes  

    pjk = data[4]
    fouls = 0
    fitness_value = 0

    for op, (S, X) in enumerate(zip(genes[::2], genes[1::2])):
        fitness_value = max(fitness_value, S + pjk[op][X - 1])

    for job_start, job_len in zip(range(0, len(genes), 2), ni):
        for op in range(job_start, job_start + (job_len - 1) * 2, 2):
            if genes[op] >= genes[op + 2]:
                fouls += 4

    for op in range(0, len(genes), 2):
        if not is_feasible_machine(op // 2, genes[op + 1], data):
            fouls += 2

    for machine in range(1, ma + 1):
        ops_on_machine = operations_in_machine(machine, genes)
        for i, op1 in enumerate(ops_on_machine):
            S1, X1 = genes[op1 * 2], pjk[op1][machine - 1]
            for op2 in ops_on_machine[i + 1:]:
                S2, X2 = genes[op2 * 2], pjk[op2][machine - 1]
                if (S2 <= S1 + X1 and S2 >= S1) or (S1 <= S2 + X2 and S1 >= S2):
                    fouls += 2

    fitness_value += fouls * 1000
    return fitness_value

steps = []

best_fitness_per_generation = []

def genetic_algorithm_scheduling(data, pop_size=100, num_generations=500):
    """
    Executes a genetic algorithm for job shop scheduling.

    Parameters:
    data (tuple): Contains information about jobs, processing times, and machine feasibility.
    pop_size (int): The size of the population for the genetic algorithm (default is 100).
    num_generations (int): The number of generations to run the algorithm (default is 500).

    Returns:
    tuple: The best individual found, including the schedule and fitness value.
    """

    start_time = time()
    ga = pyeasyga.GeneticAlgorithm(
        data, maximise_fitness=False, population_size=pop_size, generations=num_generations, mutation_probability=0.3
    )
    ga.create_individual = create_individual
    ga.mutate_function = mutate
    ga.fitness_function = lambda ind, _: fitness(ind, data)

    if len(ga.current_generation) == 0:
        ga.create_first_generation()

    for generation in range(ga.generations):
        ga.create_next_generation()
        best_fitness = min(ga.fitness_function(ind, data) for ind in ga.current_generation)
        best_fitness_per_generation.append(best_fitness)

    ga.run()
    best_individual = ga.best_individual()
    end_time = time()

    steps.append((best_individual[1], best_individual[0], end_time - start_time))
    return best_individual


result = genetic_algorithm_scheduling(data)

# Print the result
print(f"Best individual: {result[1]}")
print(f"Fitness: {result[0]}")

# Plot the best fitness over generations
plt.plot(best_fitness_per_generation)
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.title("Best Fitness Over Generations")
plt.show()


