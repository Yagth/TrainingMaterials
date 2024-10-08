import random


class NQueens:
    def __init__(self, n):
        self.n = n  # Number of queens (and the size of the chessboard)

    def fitness(self, individual):
        """
        Calculate the fitness (number of attacking pairs of queens) for the given individual.
        The cost is the number of pairs of queens that can attack each other.

        :param individual: A list representing the positions of queens in each column.
        :return: A tuple of (fitness_score, cost)
        """
        attacking_pairs = 0

        # Check for attacking pairs of queens
        for i in range(self.n):
            for j in range(i + 1, self.n):
                # Check if they are in the same row or diagonals
                if individual[i] == individual[j] or abs(individual[i] - individual[j]) == j - i:
                    attacking_pairs += 1

        # Cost is the number of attacking pairs
        cost = attacking_pairs
        # Fitness score is the inverse of the cost (lower cost means better fitness)
        fitness_score = 1 / (1 + cost)
        return fitness_score, cost

    def crossover(self, parent1, parent2):
        """
        Perform crossover between two parent individuals.

        :param parent1: First parent individual
        :param parent2: Second parent individual
        :return: Two child individuals
        """
        size = len(parent1)
        crossover_point = random.randint(0, size - 1)

        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]

        # Repair the children to ensure there are no duplicates
        child1 = self.repair(child1)
        child2 = self.repair(child2)

        return child1, child2

    def mutation(self, individual, mutation_rate):
        """
        Mutate an individual by randomly changing the position of a queen.

        :param individual: The individual to mutate
        :param mutation_rate: The probability of mutation
        :return: Mutated individual
        """
        for i in range(self.n):
            if random.random() < mutation_rate:
                # Randomly change the position of the queen in column i
                individual[i] = random.randint(0, self.n - 1)
        # Repair the individual after mutation
        return self.repair(individual)

    def selection(self, population, fitness_scores):
        """
        Perform selection to choose a parent based on fitness scores.

        :param population: The current population of individuals
        :param fitness_scores: The fitness scores for the population
        :return: Selected parent individual
        """
        total_fitness = sum(fitness_scores)
        pick = random.uniform(0, total_fitness)
        current = 0
        for i, fitness in enumerate(fitness_scores):
            current += fitness
            if current > pick:
                return population[i]

    def individual_generator(self):
        """
        Generate a random individual.

        :return: A random individual representing the positions of queens
        """
        return [random.randint(0, self.n - 1) for _ in range(self.n)]

    def repair(self, individual):
        """
        Repair the individual to ensure that no two queens are in the same row.

        :param individual: The individual to repair
        :return: The repaired individual
        """
        seen = set()
        for i in range(self.n):
            if individual[i] in seen:
                # Find a new row for this queen
                for row in range(self.n):
                    if row not in seen:
                        individual[i] = row
                        break
            seen.add(individual[i])
        return individual
