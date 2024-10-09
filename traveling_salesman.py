import math
import random

class TravelingSalesman:
    def __init__(self, points, population_size=10000, generations=100, mutation_rate=0.009):
        # Initialize the solver with poitns, population size, number of generations, and mutation rate
        self.points = points
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate

    def solve(self):
        # Generate initial random paths
        initial_generation = self.generate_population()
        current_generation = initial_generation

        # Evolve the population over a number of generations
        for _ in range(self.generations):
            current_generation = self.generate_new_population(current_generation)

        # Return the best path found
        return self.choose_best(current_generation)

    def generate_population(self):
        # Generate a list of random paths (permutations of cities)
        random_paths = []
        for _ in range(self.population_size):
            random_path = list(range(1, len(self.points)))
            random.shuffle(random_path)
            random_path = [0] + random_path  # to start from city 0
            random_paths.append(random_path)
        return random_paths

    def total_distance(self, path):
        # Calculate the total distance of the path
        return sum(math.dist(self.points[path[i - 1]], self.points[path[i]]) for i in range(len(path)))

    def choose_survivors(self, old_generation):
        # Select survivors from the old generation using a tournament selection method
        survivors = []
        random.shuffle(old_generation)  # Shuffle the generation
        midway = len(old_generation) // 2
        for i in range(midway):
            if self.total_distance(old_generation[i]) < self.total_distance(old_generation[i + midway]):
                survivors.append(old_generation[i])
            else:
                survivors.append(old_generation[i + midway])
        return survivors

    def create_children(self, parent_a, parent_b):
        # Create children by combining parts of two parents
        children = []
        start = random.randint(0, len(parent_a) - 1)
        finish = random.randint(start, len(parent_a))
        sub_path_from_a = parent_a[start:finish]
        remaining_path_from_b = list([item for item in parent_b if item not in sub_path_from_a])
        for i in range(len(parent_a)):
            if start <= i < finish:
                children.append(sub_path_from_a.pop(0))
            else:
                children.append(remaining_path_from_b.pop(0))
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
        for path in generation:
            if random.random() < self.mutation_rate:  # mutation_rate used
                index1, index2 = random.randint(1, len(path) - 1), random.randint(1, len(path) - 1)
                path[index1], path[index2] = path[index2], path[index1]  # Swap two cities
            mutated_generation.append(path)
        return mutated_generation

    def generate_new_population(self, old_generation):
        # Generate a new population by selecting survivors, applying crossovers, and mutations
        survivors = self.choose_survivors(old_generation)
        crossovers = self.apply_crossovers(survivors)
        new_population = self.apply_mutations(crossovers)
        return new_population

    def choose_best(self, paths):
        # Choose the best path with the shortest distance
        return min(paths, key=self.total_distance)

# Usage
if __name__ == "__main__":
    points = [
        (0, 0),  # City 0
        (1, 2),  # City 1
        (2, 4),  # City 2
        (3, 1),  # City 3
        (5, 3),  # City 4
        (4, 3),  # City 5
    ]

    solver = TravelingSalesman(points)
    best_path = solver.solve() 

    print(f"Best Path: {best_path} with distance {solver.total_distance(best_path)}")

