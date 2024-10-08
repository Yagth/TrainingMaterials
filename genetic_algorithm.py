import random

'''
This is a general class that helps us to implement a genetic algorithm(metaheuristic methods). 
It includes the following features of the genetic algorithm:
    1. crossover
    2. mutation
    3. elitism
    4. selection
'''


class GeneticAlgorithm:
    def __init__(self, population_size, crossover_rate, mutation_rate, generations, fitness_fn, crossover_fn, mutation_fn, selection_fn, elitism=True):
        """
        Initialize the genetic algorithm with required parameters.

        :param population_size: Number of individuals in the population.
        :param crossover_rate: Probability of crossover between two individuals.
        :param mutation_rate: Probability of mutation in an individual.
        :param generations: Number of generations to evolve.
        :param fitness_fn: Problem-specific function to calculate fitness.
        :param crossover_fn: Problem-specific function to perform crossover between two parents.
        :param mutation_fn: Problem-specific function to mutate an individual.
        :param selection_fn: Problem-specific function to select parents from the population.
        :param elitism: Boolean flag to enable/disable elitism.
        """
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.fitness_fn = fitness_fn
        self.crossover_fn = crossover_fn
        self.mutation_fn = mutation_fn
        self.selection_fn = selection_fn
        self.elitism = elitism
        self.population = []

    def initialize_population(self, individual_generator_fn):
        """
        Initialize population using a specific function to generate individuals.

        :param individual_generator_fn: Function to generate a random individual.
        """
        self.population = [individual_generator_fn()
                           for _ in range(self.population_size)]

    def evolve(self):
        """
        Main loop of the genetic algorithm to evolve the population over generations.
        """
        for generation in range(self.generations):
            # Evaluate fitness and cost for each individual in the population
            fitness_and_cost = [self.fitness_fn(individual)
                                for individual in self.population]
            fitness_scores = [fc[0]
                              for fc in fitness_and_cost]  # Extract fitness values
            costs = [fc[1]
                     for fc in fitness_and_cost]           # Extract costs

            # Track the best two individuals (elitism)
            if self.elitism:
                sorted_population = sorted(
                    self.population, key=self.fitness_fn, reverse=True)
                best_two = sorted_population[:2]
            else:
                best_two = []

            # Create a new population
            new_population = best_two  # Start with the best two individuals

            while len(new_population) < self.population_size:
                # Selection (get two parents)
                parent1 = self.selection_fn(self.population, fitness_scores)
                parent2 = self.selection_fn(self.population, fitness_scores)

                # Crossover
                if random.random() < self.crossover_rate:
                    child1, child2 = self.crossover_fn(parent1, parent2)
                else:
                    child1, child2 = parent1, parent2

                # Mutation
                child1 = self.mutation_fn(child1, self.mutation_rate)
                child2 = self.mutation_fn(child2, self.mutation_rate)

                # Add offspring to new population
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)

            self.population = new_population

            # Optionally track the best individual and cost
            best_fitness = max(fitness_scores)
            best_individual = self.population[fitness_scores.index(
                best_fitness)]
            best_cost = costs[fitness_scores.index(best_fitness)]

            print(
                f"Generation {generation}: Best fitness {best_fitness}, Cost {best_cost} with individual {best_individual}")

        # Return the best individual, fitness, and cost after all generations
        best_fitness = max(fitness_scores)
        best_individual = self.population[fitness_scores.index(best_fitness)]
        best_cost = costs[fitness_scores.index(best_fitness)]
        return best_individual, best_fitness, best_cost
