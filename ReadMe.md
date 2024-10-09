# Generic Programming Algorithms: TSP, JSSP, and N-Queens

This project implements three classical combinatorial optimization problems using genetic algorithms and other heuristic approaches:

    Traveling Salesman Problem (TSP)
    Job Shop Scheduling Problem (JSSP)
    N-Queens Problem


## Project Structure

    tsp.py: This script implements the TSP using a genetic algorithm.
    jssp.py: This script tackles the JSSP using genetic heuristics for job sequencing and machine scheduling.
    n_queens.py: This script solves the N-Queens problem using backtracking.
    tsp.txt: A text file containing the coordinates of cities for the TSP.

## Problem Descriptions and Solutions
1. Travelling Salesman Problem (TSP)

The objective is to find the shortest possible route that visits each city exactly once and returns to the starting city. The solution uses:

    Crossover: Combines two parent routes to create a child route.
    Mutation: Randomly swaps cities in the route to explore new routes.
    Elitism: Keeps the best routes for the next generation.
    Selection: Selects parents using tournament selection to breed the next generation.

Main Functions:

    getCity(): Loads city data from a file (tsp.txt).
    calcDistance(): Calculates the total distance of a route.
    selectPopulation(): Creates an initial population of routes.
    geneticAlgorithm(): The main genetic algorithm to evolve better routes.
    drawMap(): Visualizes the cities and the final route.

2. Job Shop Scheduling Problem (JSSP)

This problem involves scheduling jobs on machines in a way that minimizes the overall completion time. A genetic algorithm is used for:

    Crossover: Exchanges job sequences between parents.
    Mutation: Swaps job sequences to explore new solutions.
    Fitness: Calculates the makespan of a job sequence, i.e., the time at which all jobs are completed.

Main Functions:

    fitness(): Evaluates a job sequence by calculating the makespan.
    selection(): Selects a subset of job sequences for crossover based on their fitness.
    crossover(): Generates new job sequences by combining parents.
    geneticAlgorithm(): Evolves a population of job sequences to minimize the makespan.

3. N-Queens Problem

The goal is to place N queens on an N×N chessboard such that no two queens threaten each other. This is a constraint satisfaction problem where the solution is achieved using:

    Genetic Representation: A chromosome represents the row positions of queens on the chessboard.
    Fitness: Evaluates the number of queens that do not attack each other.

The solution for the N-Queens problem is structured similarly to TSP and JSSP, using genetic operators to evolve valid solutions.
Usage
Requirements

    Python 3.x
    numpy
    matplotlib (for visualizing the TSP solution)

Install dependencies using:
    pip install numpy matplotlib
Running the Code

    Travelling Salesman Problem:
        Make sure to place tsp.txt (containing city coordinates) in the same directory as the script.
        Run the tsp_algorithm.py file:

        bash

    python tsp_algorithm.py

Job Shop Scheduling Problem:

    Run the jssp_algorithm.py file:

    python jssp_algorithm.py

N-Queens Problem:
    Update the problem size (N) in the n_queens_algorithm.py file and run the script:

       python n_queens_algorithm.py

Parameters

You can tune the following parameters for the genetic algorithms:

    POPULATION_SIZE: Number of individuals in each generation.
    MUTATION_RATE: Probability of mutation in a chromosome.
    CROSSOVER_RATE: Probability of crossover between two parent chromosomes.
    TARGET: The objective value to stop the algorithm (e.g., minimal distance for TSP).

Visualization

For the TSP solution, the cities and the best route will be visualized using matplotlib.