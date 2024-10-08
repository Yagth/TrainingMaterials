# ---- Example Usage of the TSPProblem class with GeneticAlgorithm ----

from traverse_selse_man.traverse_selse_genetic import TSPProblem
from genetic_algorithm import GeneticAlgorithm


graph = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
]

# Instantiate the TSP problem
tsp_problem = TSPProblem(graph)

# Initialize the GA with TSP-specific utility functions and elitism enabled
ga = GeneticAlgorithm(
    population_size=100,
    crossover_rate=0.9,
    mutation_rate=0.02,
    generations=500,
    fitness_fn=tsp_problem.fitness,
    crossover_fn=tsp_problem.crossover,
    mutation_fn=tsp_problem.mutate,
    selection_fn=tsp_problem.selection,
    elitism=True  # Enable elitism
)

ga.initialize_population(tsp_problem.generate_individual)
best_individual, best_fitness, best_cost = ga.evolve()

print(f"\nBest path found: {best_individual}")
print(f"Best(shortest length): {best_cost}")
print(f"Best fitness: {best_fitness}")
