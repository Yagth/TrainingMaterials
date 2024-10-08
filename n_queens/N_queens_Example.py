# Number of queens
from genetic_algorithm import GeneticAlgorithm
from n_queens.n_queens_genetic import NQueens


n = 8

# Initialize the NQueens problem
n_queens = NQueens(n)

# Initialize the Genetic Algorithm with NQueens-specific utility functions
ga = GeneticAlgorithm(
    population_size=100,
    crossover_rate=0.9,
    mutation_rate=0.02,
    generations=500,
    fitness_fn=n_queens.fitness,
    crossover_fn=n_queens.crossover,
    mutation_fn=n_queens.mutation,
    selection_fn=n_queens.selection,
    elitism=True  # Enable elitism
)

ga.initialize_population(n_queens.individual_generator)
best_individual, best_fitness, best_cost = ga.evolve()

print(f"\nBest individual found: {best_individual}")
print(f"Cost (attacking pairs): {best_cost}")
print(f"Fitness score: {best_fitness}")
