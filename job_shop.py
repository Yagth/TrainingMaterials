import random
from typing import List, Tuple

Job = List[Tuple[int, int]]  # machine id, processing time tuples
Schedule = List[List[int]]  # List of job indices for each machine

class JobShop:
    def __init__(self, jobs, population_size=50, generations=1000, mutation_rate=0.1):
        # Initialize the solver with jobs, population size, number of generations, and mutation rate
        self.jobs = jobs
        self.num_jobs = len(jobs)
        self.num_machines = len(jobs[0])
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate

    def solve(self):
        # Generate initial random population
        initial_population = self.generate_population()
        current_population = initial_population

        # Evolve the population over a number of generations
        for _ in range(self.generations):
            current_population = self.generate_new_population(current_population)

        # Return the best schedule found
        return self.choose_best(current_population)

    def generate_random_schedule(self):
        # Generate a random schedule for all jobs and machines
        schedule = [[] for _ in range(self.num_machines)]
        job_order = list(range(self.num_jobs))
        random.shuffle(job_order)
        for job in job_order:
            for machine in range(self.num_machines):
                schedule[machine].append(job)
        return schedule

    def generate_population(self):
        # Generate a population of random schedules
        return [self.generate_random_schedule() for _ in range(self.population_size)]

    def fitness(self, schedule):
        # Calculate the fitness of the schedule (inverse of makespan)
        makespan = self.calculate_makespan(schedule)
        return 1 / (makespan + 1)  # adding 1 to avoid division by zero

    def calculate_makespan(self, schedule):
        # Calculate the makespan (total time) of the schedule
        machine_times = [0] * self.num_machines
        job_completion_times = [0] * self.num_jobs

        for machine_id, machine_schedule in enumerate(schedule):
            for job_id in machine_schedule:
                job = self.jobs[job_id]
                machine_time = job[machine_id][1]
                start_time = max(machine_times[machine_id], job_completion_times[job_id])
                machine_times[machine_id] = start_time + machine_time
                job_completion_times[job_id] = machine_times[machine_id]

        return max(machine_times)

    def choose_survivors(self, population):
        # Select survivors using a tournament selection method
        survivors = []
        random.shuffle(population)
        midway = len(population) // 2
        for i in range(midway):
            if self.fitness(population[i]) > self.fitness(population[i + midway]):
                survivors.append(population[i])
            else:
                survivors.append(population[i + midway])
        return survivors

    def create_children(self, parent_a, parent_b):
        # Create children by combining parts of two parent schedules
        children = [[] for _ in range(self.num_machines)]
        for machine in range(self.num_machines):
            crossover_point = random.randint(0, len(parent_a[machine]))
            children[machine] = parent_a[machine][:crossover_point] + parent_b[machine][crossover_point:]
        return children

    def apply_crossovers(self, survivors):
        # Apply crossover to generate new children from survivors
        children = []
        midway = len(survivors) // 2
        for i in range(midway):
            parent_a, parent_b = survivors[i], survivors[i + midway]
            for _ in range(2):  # Each pair produces two children
                children.append(self.create_children(parent_a, parent_b))
                children.append(self.create_children(parent_b, parent_a))
        return children

    def apply_mutations(self, generation):
        # Apply mutations to the generation with a given mutation rate
        mutated_generation = []
        for schedule in generation:
            if random.random() < self.mutation_rate:
                for machine in schedule:
                    i, j = random.sample(range(len(machine)), 2)
                    machine[i], machine[j] = machine[j], machine[i]  # Swap two jobs
            mutated_generation.append(schedule)
        return mutated_generation

    def generate_new_population(self, old_population):
        # Generate a new population by selecting survivors, applying crossovers, and mutations
        survivors = self.choose_survivors(old_population)
        crossovers = self.apply_crossovers(survivors)
        new_population = self.apply_mutations(crossovers)
        return new_population

    def choose_best(self, population):
        # Choose the best schedule with the highest fitness
        return max(population, key=self.fitness)

# Example usage
if __name__ == "__main__":
    # Define a simple job shop problem
    jobs = [
        [(0, 3), (1, 2), (2, 2)],  # Job 0
        [(0, 2), (2, 1), (1, 4)],  # Job 1
        [(1, 4), (2, 3), (0, 3)],  # Job 2
    ]

    solver = JobShop(jobs)
    best_schedule = solver.solve()  # Get the best schedule
    best_makespan = solver.calculate_makespan(best_schedule)
    best_fitness = solver.fitness(best_schedule)

    print(f"Best schedule found:")
    for machine_id, machine_schedule in enumerate(best_schedule):
        print(f"Machine {machine_id}: {machine_schedule}")
    print(f"Makespan: {best_makespan}")
    print(f"Fitness: {best_fitness}")