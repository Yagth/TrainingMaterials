# Genetic Algorithm on Python
This project implements Genetic Algorithms (GA) to solve three classic optimization problems: Job Shop Scheduling, N-Queens, and the Traveling Salesman Problem (TSP). Each of these problems is complex and involves finding an optimal or near-optimal solution through evolutionary techniques.

## Requirement
```
pip install matplotlib
pip install pyeasyga
```

## Explanation of Algorithm
Flowchart of GA

![Flowchart of GA](https://cdn-images-1.medium.com/max/1600/1*HP8JVxlJtOv14rGLJfXEzA.png)

### 1. Initialize Population
> What is population? Population is a collection of genes

> What is gen? its an individual in the population

In the TSP, the population is a set of possible routes (tours) for visiting all the cities. This is similar to creating random strings in the text matching problem. Each individual in the population is a route, and each route's fitness is determined by the total distance covered.
```
# Create a population of random routes (tours)
def create_initial_population(cities, population_size):
    population = []
    for _ in range(population_size):
        tour = cities.copy()
        random.shuffle(tour)
        distance = calculate_total_distance(tour)
        population.append((distance, tour))
    fittest = min(population, key=lambda x: x[0])
    return population, fittest
```
In this step, the population consists of random tours, and the fitness of each tour is the total distance of that tour. In the text GA, the fitness was based on the similarity to the target string.

### 2. Selection

> Selection is a process to choose 2 best from a population

The selection process remains almost the same, where two "parents" are selected from the population. In this case, we use tournament selection, picking two individuals based on their fitness (distance) and selecting the ones with the lowest distance:
```
# Select two fittest routes from a tournament of random individuals
def tournament_selection(population, selection_size):
    return min(random.choices(population, k=selection_size), key=lambda x: x[0])
```

### 3. Crossover

> In the TSP, crossover means combining two routes to create new offspring (routes). The idea is to take part of one parent route and combine it with part of the other parent's route, ensuring no
city is repeated.

In this code, crossover process is making process like this :
```
# Perform crossover between two parents to create two offspring
def crossover(parent1, parent2, crossover_rate, len_cities):
    if random.random() < crossover_rate:
        point = random.randint(0, len_cities - 1)
        child1 = parent1[1][:point]
        child2 = parent2[1][:point]
        child1 += [city for city in parent2[1] if city not in child1]
        child2 += [city for city in parent1[1] if city not in child2]
    else:
        child1 = parent1[1].copy()
        child2 = parent2[1].copy()
    return child1, child2
```

### 4. Mutation

> mutation process is genetic operator used to maintain genetic diversity from one generation of a population of genetic algorithm chromosomes to the next

In the mutation step, we randomly swap two cities in the route to introduce variety in the population, much like how characters are randomly altered in the text matching GA.

```
# Randomly swap two cities in the tour based on the mutation rate
def mutate(tour, mutation_rate, len_cities):
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(len_cities), 2)
        # Swap two cities
        tour[idx1], tour[idx2] = tour[idx2], tour[idx1]
```

### 5. Evaluation

> evaluation process is check what fitness value

Just like in the text GA where fitness was calculated to check if the guessed string matched the target, here we calculate the total distance of the mutated (or offspring) tours to evaluate their fitness.

```
# Calculate the total distance (fitness) of a tour
def calculate_total_distance(tour):
    total_distance = 0
    for i in range(len(tour) - 1):
        total_distance += calculate_distance(tour[i], tour[i + 1])
    total_distance += calculate_distance(tour[-1], tour[0])  # Return to the starting city
    return total_distance
```

### 6. Regeneration of Population

> regeneration is insert gen of mutation process into a population

After creating new offspring through crossover and mutation, we insert these new individuals into the population, replacing the worst-performing ones. This keeps the population evolving towards better solutions.

```
# Evolve the population by generating offspring and replacing the least fit
def evolve_population(population, len_cities, selection_size, mutation_rate, crossover_rate, target_distance, max_generations=150):
    generation_number = 0
    for _ in range(max_generations):
        # Keep the two best routes from the current population
        new_population = sorted(population)[:2]  
        while len(new_population) < len(population):
            # Select parents via tournament selection
            parent1 = tournament_selection(population, selection_size)
            parent2 = tournament_selection(population, selection_size)
            # Generate offspring via crossover
            child1, child2 = crossover(parent1, parent2, crossover_rate, len_cities)
            # Mutate the offspring
            mutate(child1, mutation_rate, len_cities)
            mutate(child2, mutation_rate, len_cities)
            # Add the new offspring to the population
            new_population.append((calculate_total_distance(child1), child1))
            new_population.append((calculate_total_distance(child2), child2))
        # Replace the old population with the new one
        population = new_population
        generation_number += 1

        # Check if the target distance is reached
        if min(population, key=lambda x: x[0])[0] < target_distance:
            break
    return min(population, key=lambda x: x[0]), generation_number
```
