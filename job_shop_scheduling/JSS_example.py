
# Example usage of the GeneticAlgorithm class with JobShopProblem
from genetic_algorithm import GeneticAlgorithm
from job_shop_scheduling.job_shop_schedu_genetic import JobShopProblem

# Sample Job-Shop problem where each job has tasks assigned to specific machines with specific processing times


jobs = [
    # Job 0 with tasks (Machine 0, time 3), (Machine 1, time 2), (Machine 2, time 2)
    [(0, 3), (1, 2), (2, 2)],
    # Job 1 with tasks (Machine 0, time 2), (Machine 2, time 1), (Machine 1, time 4)
    [(0, 2), (2, 1), (1, 4)],
    # Job 2 with tasks (Machine 1, time 4), (Machine 2, time 3), (Machine 0, time 2)
    [(1, 4), (2, 3), (0, 2)]
]

num_machines = 3

# Instantiate the JSS problem
jss_problem = JobShopProblem(jobs, num_machines)

# Instantiate the Genetic Algorithm with problem-specific functions from JSS

ga = GeneticAlgorithm(
    population_size=10,
    crossover_rate=0.7,
    mutation_rate=0.1,
    generations=50,
    fitness_fn=jss_problem.fitness,
    crossover_fn=jss_problem.crossover,
    mutation_fn=jss_problem.mutate,
    selection_fn=jss_problem.selection,
    elitism=True
)

# Initialize population and evolve
ga.initialize_population(jss_problem.generate_individual)
best_individual, best_fitness, best_cost = (ga.evolve())

print("Best schedule:", best_individual)
print("Best makespan (minimum total completion time):", best_fitness)
