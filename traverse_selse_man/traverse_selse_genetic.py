import random
from genetic_algorithm import GeneticAlgorithm


class TSPProblem:
    def __init__(self, graph):
        """
        Initialize the TSP problem.

        :param graph: A 2D list representing the distance between cities. 
                      graph[i][j] is the distance from city i to city j.
        """
        self.graph = graph
        self.n = len(graph)  # Number of cities

    def fitness(self, path):
        """
        Calculate the fitness of a given path. Fitness is the inverse of the total distance.

        :param path: A list representing a path through all cities.
        :return: A tuple (fitness, total distance). Fitness is the inverse of distance.
        """
        distance = 0
        for i in range(len(path) - 1):
            distance += self.graph[path[i]][path[i + 1]]
        distance += self.graph[path[-1]][path[0]]  # Return to the start
        return 1 / distance, distance  # Inverse of distance as fitness

    def crossover(self, parent1, parent2):
        """
        Perform crossover between two parent paths. 
        Use a Partially Mapped Crossover (PMX)-like approach.

        :param parent1: The first parent path.
        :param parent2: The second parent path.
        :return: Two offspring paths.
        """
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        child = [-1] * size

        child[start:end] = parent1[start:end]
        pointer = end
        for city in parent2:
            if city not in child:
                if pointer == size:
                    pointer = 0
                child[pointer] = city
                pointer += 1
        return child, child  # Return two identical children

    def mutate(self, path, mutation_rate):
        """
        Mutate the path by swapping two cities with a certain probability.

        :param path: The path to be mutated.
        :param mutation_rate: The probability of mutation.
        :return: The mutated path.
        """
        for i in range(len(path)):
            if random.random() < mutation_rate:
                j = random.randint(0, len(path) - 1)
                path[i], path[j] = path[j], path[i]
        return path

    def selection(self, population, fitness_scores):
        """
        Select an individual from the population using roulette wheel selection.

        :param population: The current population.
        :param fitness_scores: The fitness scores for the population.
        :return: A selected parent individual.
        """
        total_fitness = sum(fitness_scores)
        pick = random.uniform(0, total_fitness)
        current = 0
        for i, fitness in enumerate(fitness_scores):
            current += fitness
            if current > pick:
                return population[i]

    def generate_individual(self):
        """
        Generate a random individual (path) for the TSP problem.

        :return: A list representing a path through all cities.
        """
        return random.sample(range(self.n), self.n)

# ---- Example Usage of the TSPProblem class with GeneticAlgorithm ----


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
