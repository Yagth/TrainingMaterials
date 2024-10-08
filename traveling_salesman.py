import random
import math


def load_cities(file_name="TSP.txt"):
    """
    Load city data from a specified file.

    Reads a file with city names and their coordinates (x, y), returning a list of cities.

    Parameters:
    -----------
    file_name : str, optional
        The name of the file to read from (default: "TSP.txt").

    Returns:
    --------
    list
        A list of cities, each represented as [city name (str), x-coordinate (float), y-coordinate (float)].

    Example:
    --------
    >>> cities = load_cities("TSP51.txt")
    >>> print(cities)
    [['CityA', 12.34, 56.78], ['CityB', 90.12, 34.56], ...]
    """

    cities = []
    with open(file_name) as f:
        for line in f.readlines():
            node_city_val = line.split()
            cities.append([node_city_val[0], float(node_city_val[1]), float(node_city_val[2])])
    return cities

def calculate_distance(cityA, cityB):
    """
    Calculate the Euclidean distance between two cities.

    Parameters:
    cityA : list
        First city with name and coordinates (x, y).
    cityB : list
        Second city with name and coordinates (x, y).

    Returns:
    float
        The distance between the two cities.

    Example:
    >>> calculate_distance(['CityA', 1.0, 2.0], ['CityB', 4.0, 6.0])
    5.0
    """

    return math.sqrt(math.pow(cityB[1] - cityA[1], 2) + math.pow(cityB[2] - cityA[2], 2))


def calculate_total_distance(tour):
    """
    Calculate the total distance of a given tour of cities.

    Parameters:
    tour : list
        A list of cities, where each city is represented by a list containing 
        its name and coordinates (x, y).

    Returns:
    float
        The total distance of the tour, including the return to the starting city.

    Example:
    >>> tour = [['CityA', 1.0, 2.0], ['CityB', 4.0, 6.0], ['CityC', 7.0, 1.0]]
    >>> calculate_total_distance(tour)
    15.0
    """

    total_distance = 0
    for i in range(len(tour) - 1):
        total_distance += calculate_distance(tour[i], tour[i + 1])

    total_distance += calculate_distance(tour[-1], tour[0])
    return total_distance


def create_initial_population(cities, population_size):
    """
    Create an initial population of randomized tours for the Traveling Salesman Problem (TSP).

    Parameters:
    cities : list
        List of cities, each represented as [name, x, y].
    population_size : int
        Number of tours to generate.

    Returns:
    tuple
        A tuple containing:
            - population : list of tuples (distance, tour)
            - fittest : tuple (fittest distance, fittest tour)
    """

    population = []
    for _ in range(population_size):
        tour = cities.copy()
        random.shuffle(tour)
        distance = calculate_total_distance(tour)
        population.append((distance, tour))
    fittest = min(population, key=lambda x: x[0])
    return population, fittest


def tournament_selection(population, selection_size):
    """
    Select the fittest individual from a random subset of the population.

    Parameters:
    population : list
        List of individuals, each represented as (distance, tour).
    selection_size : int
        Number of individuals to randomly select for the tournament.

    Returns:
    tuple
        The fittest individual from the selected subset, represented as (distance, tour).
    """

    return min(random.choices(population, k=selection_size), key=lambda x: x[0])


def crossover(parent1, parent2, crossover_rate, len_cities):
    """
    Performs crossover between two parents in a genetic algorithm to generate two children.

    Args:
        parent1 (tuple): A tuple with an identifier and a list of cities.
        parent2 (tuple): Another tuple similar to `parent1`.
        crossover_rate (float): Probability of crossover (between 0 and 1).
        len_cities (int): Total number of cities in the route.

    Returns:
        tuple: Two lists, `child1` and `child2`, representing the offspring.
    """
    
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


def mutate(tour, mutation_rate, len_cities):
    """
    Randomly swaps two cities in the tour based on the mutation rate.

    Parameters:
    tour : list
        The current tour represented as a list of cities.
    mutation_rate : float
        Probability of mutation occurring (between 0 and 1).
    len_cities : int
        The total number of cities in the tour.

    Returns:
    None
    """

    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(len_cities), 2)
        tour[idx1], tour[idx2] = tour[idx2], tour[idx1]


def evolve_population(population, len_cities, selection_size, mutation_rate, crossover_rate, target_distance, max_generations=150):
    """
    Evolve the population of tours using genetic algorithm principles.

    Parameters:
    population : list
        Current population of tours, each represented as a tuple (distance, tour).
    len_cities : int
        The total number of cities in the tours.
    selection_size : int
        Number of individuals to select for tournament selection.
    mutation_rate : float
        Probability of mutation occurring for each offspring (between 0 and 1).
    crossover_rate : float
        Probability of crossover occurring between parent tours (between 0 and 1).
    target_distance : float
        The desired target distance for early termination.
    max_generations : int, optional
        Maximum number of generations to evolve (default is 150).

    Returns:
    tuple
        The fittest tour (distance, tour) and the number of generations evolved.
    """

    generation_number = 0
    for _ in range(max_generations):
        new_population = sorted(population)[:2]  

        while len(new_population) < len(population):
            parent1 = tournament_selection(population, selection_size)
            parent2 = tournament_selection(population, selection_size)
            child1, child2 = crossover(parent1, parent2, crossover_rate, len_cities)
            
            mutate(child1, mutation_rate, len_cities)
            mutate(child2, mutation_rate, len_cities)
            
            new_population.append((calculate_total_distance(child1), child1))
            new_population.append((calculate_total_distance(child2), child2))

        population = new_population
        generation_number += 1

        if generation_number % 15 == 0:
            print(f"Generation {generation_number}: Best Distance = {min(population, key=lambda x: x[0])[0]}")

        if min(population, key=lambda x: x[0])[0] < target_distance:
            break

    return min(population, key=lambda x: x[0]), generation_number


def visualize_route(cities, best_tour):
    """
    Visualize the route of the best tour on a 2D plot.

    Parameters:
    cities : list
        A list of cities, where each city is represented as a list containing
        [city name, x-coordinate, y-coordinate].
    best_tour : tuple
        The best tour found, represented as a tuple (distance, tour),
        where 'tour' is a list of cities in the order they are visited.

    Returns:
    None
    """

    import matplotlib.pyplot as plt

    for city in cities:
        plt.plot(city[1], city[2], "ro")
        plt.annotate(city[0], (city[1], city[2]))

    for i in range(len(best_tour[1]) - 1):
        cityA = best_tour[1][i]
        cityB = best_tour[1][i + 1]
        plt.plot([cityA[1], cityB[1]], [cityA[2], cityB[2]], "gray")

    cityA = best_tour[1][0]
    cityB = best_tour[1][-1]
    plt.plot([cityA[1], cityB[1]], [cityA[2], cityB[2]], "gray")

    plt.show()


def main():
    """
    Main function to execute the Genetic Algorithm for solving the Traveling Salesman Problem (TSP).

    This function initializes the parameters for the genetic algorithm, loads the city data,
    creates the initial population, evolves the population through selection, crossover, and mutation,
    and visualizes the best tour found.

    The parameters for the genetic algorithm include:
    - POPULATION_SIZE: Number of tours in the population.
    - SELECTION_SIZE: Number of parents selected for tournament selection.
    - MUTATION_RATE: Probability of mutating a tour during evolution.
    - CROSSOVER_RATE: Probability of crossing over two parent tours to create offspring.
    - TARGET_DISTANCE: The desired distance threshold to stop the evolution process.

    The function prints out the results of the genetic algorithm, including:
    - The initial fittest distance from the initial population.
    - The final fittest distance after evolution.
    - The number of generations taken to reach the final solution.
    - The target distance set for the algorithm.

    Finally, it visualizes the best tour found using the visualize_route function.

    Returns:
    None
    """

    POPULATION_SIZE = 2500
    SELECTION_SIZE = 3
    MUTATION_RATE = 0.1
    CROSSOVER_RATE = 0.9
    TARGET_DISTANCE = 250.0

    cities = load_cities()
    initial_population, initial_fittest = create_initial_population(cities, POPULATION_SIZE)
    
    best_solution, generations = evolve_population(
        initial_population,
        len(cities),
        SELECTION_SIZE,
        MUTATION_RATE,
        CROSSOVER_RATE,
        TARGET_DISTANCE
    )

    print("\n------------------ Genetic Algorithm Results ------------------")
    print(f"Initial Fittest Distance: {initial_fittest[0]}")
    print(f"Final Fittest Distance: {best_solution[0]}")
    print(f"Generations Taken: {generations}")
    print(f"Target Distance: {TARGET_DISTANCE}")
    print("---------------------------------------------------------------\n")

    visualize_route(cities, best_solution)


if __name__ == "__main__":
    main()


