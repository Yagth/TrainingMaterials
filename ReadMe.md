# Optimization Problems Solved with Genetic Algorithms

This repository contains implementations of three classic optimization problems solved using genetic algorithms:

1. Traveling Salesman Problem (TSP)
2. Job Shop Scheduling Problem
3. N-Queens Problem

## 1. Traveling Salesman Problem (traveling_salesman.py)

The Traveling Salesman Problem is a classic optimization problem where the goal is to find the shortest possible route that visits each city exactly once and returns to the starting city.

### Features:
- Implements a genetic algorithm to solve the TSP
- Uses tournament selection for choosing survivors
- Applies crossover and mutation operations to generate new populations
- Calculates the total distance of a path

## 2. Job Shop Scheduling Problem (job_shop.py)

The Job Shop Scheduling Problem involves scheduling a set of jobs on a set of machines, with each job consisting of a sequence of operations that must be processed in a specific order.

### Features:
- Implements a genetic algorithm to solve the Job Shop Scheduling Problem
- Generates random schedules for initial population
- Uses fitness function based on makespan (total completion time)
- Applies crossover and mutation operations to evolve schedules

## 3. N-Queens Problem (nQueen.py)

The N-Queens Problem is the challenge of placing N chess queens on an N×N chessboard so that no two queens threaten each other.

### Features:
- Implements a genetic algorithm to solve the N-Queens Problem
- Generates random queen arrangements for initial population
- Uses a fitness function based on non-attacking pairs of queens
- Applies crossover and mutation operations to evolve solutions

## Usage

Each problem is implemented in its own Python file. To run a specific problem, execute the corresponding Python script aftering cloning the repository. For example:

```bash
python traveling_salesman.py
python job_shop.py
python nQueen.py
```

This will execute the genetic algorithm for each problem and display the results.

