import random

class NQueens:
    def __init__(self, n, population_size=100, generations=1000, mutation_rate=0.01):
        # Initialize the solver with board size, population size, number of generations, and mutation rate
        self.n = n
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate

    def solve(self):
        # Generate initial random population
        initial_generation = self.generate_population()
        current_generation = initial_generation

        # Evolve the population over a number of generations
        for _ in range(self.generations):
            current_generation = self.generate_new_population(current_generation)

        # Return the best solution found
        return self.choose_best(current_generation)

    def generate_population(self):
        # Generate a list of random arrangements of queens
        random_solutions = []
        for _ in range(self.population_size):
            solution = list(range(self.n))
            random.shuffle(solution)
            random_solutions.append(solution)
        return random_solutions

    def fitness(self, solution):
        # Calculate the number of non-attacking pairs of queens
        n = len(solution)
        non_attacking_pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                if solution[i] != solution[j] and abs(solution[i] - solution[j]) != j - i:
                    non_attacking_pairs += 1
        return non_attacking_pairs

    def choose_survivors(self, old_generation):
        # Select survivors from the old generation using a tournament selection method
        survivors = []
        random.shuffle(old_generation)
        midway = len(old_generation) // 2
        for i in range(midway):
            if self.fitness(old_generation[i]) > self.fitness(old_generation[i + midway]):
                survivors.append(old_generation[i])
            else:
                survivors.append(old_generation[i + midway])
        return survivors

    def create_children(self, parent_a, parent_b):
        # Create children by combining parts of two parents
        n = len(parent_a)
        start = random.randint(0, n - 1)
        finish = random.randint(start, n)
        sub_path_from_a = parent_a[start:finish]
        remaining_path_from_b = [item for item in parent_b if item not in sub_path_from_a]
        children = sub_path_from_a + remaining_path_from_b
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
        for solution in generation:
            if random.random() < self.mutation_rate:
                i, j = random.sample(range(self.n), 2)
                solution[i], solution[j] = solution[j], solution[i]  # Swap two queens
            mutated_generation.append(solution)
        return mutated_generation

    def generate_new_population(self, old_generation):
        # Generate a new population by selecting survivors, applying crossovers, and mutations
        survivors = self.choose_survivors(old_generation)
        crossovers = self.apply_crossovers(survivors)
        new_population = self.apply_mutations(crossovers)
        return new_population

    def choose_best(self, solutions):
        # Choose the best solution with the highest fitness
        return max(solutions, key=self.fitness)

# Usage
if __name__ == "__main__":
    n = 8  # Number of queens
    solver = NQueens(n)
    best_solution = solver.solve()

    print(f"Best Solution: {best_solution} with fitness {solver.fitness(best_solution)}")
