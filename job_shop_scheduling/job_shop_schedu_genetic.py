import random
import copy


# It handles the core logic of crossover, mutation, and selection.


class JobShopProblem:
    def __init__(self, jobs, machines):
        """
        Initialize the Job-Shop Scheduling problem. 

        :param jobs: A list of jobs, where each job is a list of tasks (machine, processing time).
        :param machines: Number of machines available in the system.
        """

        self.jobs = jobs
        self.machines = machines
        self.num_jobs = len(jobs)
        self.num_tasks = sum(len(job) for job in jobs)

    def generate_individual(self):
        """
        Generate a random individual for the JSS problem.
        An individual is represented as a sequence of tasks in a random order.
        """
        task_sequence = [(job_index, task_index) for job_index in range(
            self.num_jobs) for task_index in range(len(self.jobs[job_index]))]
        random.shuffle(task_sequence)
        return task_sequence

    def fitness(self, individual):
        """
        Calculate the fitness (makespan) and total cost for the given individual.

        :param individual: A sequence of tasks represented as (job_index, task_index) tuples.
        :return: A tuple (makespan, total_cost) where makespan is the maximum time across machines,
                and total_cost is the sum of processing times for all tasks.
        """
        # Each machine has a current time tracker

        machine_times = [0] * self.machines
        job_times = [0] * self.num_jobs
        total_cost = 0

        for job_idx, task_idx in individual:
            machine, time_needed = self.jobs[job_idx][task_idx]

            # A task starts when both the machine and the job are available
            start_time = max(machine_times[machine], job_times[job_idx])
            machine_times[machine] = start_time + time_needed
            # Update job's availability
            job_times[job_idx] = machine_times[machine]

            # Add the time needed for the task to the total cost
            total_cost += time_needed

        # Makespan is the maximum time among all machines
        makespan = max(machine_times)

        # Return both makespan (fitness) and total cost
        return makespan, total_cost

    def crossover(self, parent1, parent2):
        """
        Perform crossover between two parent individuals.
        This will implement a uniform crossover for simplicity.

        :param parent1: The first parent individual.
        :param parent2: The second parent individual.
        :return: Two offspring individuals.
        """
        cut = random.randint(0, len(parent1) - 1)
        child1 = parent1[:cut] + \
            [task for task in parent2 if task not in parent1[:cut]]
        child2 = parent2[:cut] + \
            [task for task in parent1 if task not in parent2[:cut]]
        return child1, child2

    def mutate(self, individual, mutation_rate):
        """
        Mutate the individual by swapping two random tasks with a certain probability.

        :param individual: The individual to be mutated.
        :param mutation_rate: The mutation probability.
        :return: The mutated individual.
        """
        mutated = copy.deepcopy(individual)
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                j = random.randint(0, len(mutated) - 1)
                mutated[i], mutated[j] = mutated[j], mutated[i]
        return mutated

    def selection(self, population, fitness_scores):
        """
        Select an individual from the population using roulette wheel selection.

        :param population: The current population.
        :param fitness_scores: The fitness scores for the population.
        :return: A selected parent individual.
        """
        total_fitness = sum(fitness_scores)
        selection_probs = [
            fitness / total_fitness for fitness in fitness_scores]
        return random.choices(population, weights=selection_probs, k=1)[0]
