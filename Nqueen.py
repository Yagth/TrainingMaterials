import numpy as np
import matplotlib.pyplot as plt
import random

def checkFitness(pop):
    """
    Calculates the fitness of each solution in the population by counting the number of diagonal conflicts between queens.

    Parameters:
    -----------
    pop : numpy.ndarray
        A 2D array where each row is a solution (arrangement of queens).

    Returns:
    --------
    fit : numpy.ndarray
        A column vector of fitness values, where each entry is the number of diagonal conflicts for a solution.
    """

    fit = np.zeros((pop[:, 1].size, 1))
    
    for index, solution in enumerate(pop):
        for ia, a in enumerate(solution, start=1):
            for ib, b in enumerate(solution[ia:], start=ia + 1):
                if abs(a - b) == abs(ia - ib):
                    fit[index, 0] += 1
    
    return fit

def order_crossover(p1, p2, size):
    """
    Performs an Order Crossover (OX) on two parent solutions to create two offspring.
    Order crossover preserves the relative order of genes between two parents.

    Parameters:
    -----------
    p1 : numpy.ndarray
        The first parent solution (array of queen positions).
    p2 : numpy.ndarray
        The second parent solution (array of queen positions).
    size : int
        The number of genes to be directly copied from each parent to their respective offspring.

    Returns:
    --------
    offsprings : numpy.ndarray
        A 2D array containing two offspring solutions created by crossover of p1 and p2.
    """

    def fillGene(f, p):
        """
        Fills the offspring with missing genes from the other parent while maintaining gene order.

        Parameters:
        -----------
        f : numpy.ndarray
            Partially filled offspring.
        p : numpy.ndarray
            The other parent providing missing genes.

        Returns:
        --------
        f : numpy.ndarray
            A fully filled offspring solution.
        """

        for ia, a in enumerate(p):
            if a not in f:
                for ib, b in enumerate(f):
                    if b == 0:  
                        f[ib] = a
                        break
        return f

    f1 = np.zeros(len(p1), dtype=int)  
    f2 = np.zeros(len(p2), dtype=int)  

    c = random.randint(0, (len(p1) - size))

    f1[c:c + size] = p1[c:c + size]
    f2[c:c + size] = p2[c:c + size]

    f1 = fillGene(f1, p2)
    f2 = fillGene(f2, p1)

    offsprings = np.vstack([f1, f2])
    return offsprings


def selection(pop, p_sel):
    """
    Performs tournament selection to choose the best solution (individual) from a population based on fitness.

    Parameters:
    -----------
    pop : numpy.ndarray
        The population of solutions (each row is a solution, and the last column contains fitness scores).
    p_sel : float
        The proportion of the population to randomly select for the tournament (a value between 0 and 1).

    Returns:
    --------
    bestSol : numpy.ndarray
        The best solution (individual) selected from the randomly chosen pool of solutions.
    """

    sel_pool = np.random.permutation(pop[:, 1].size)[0:int(round(pop[:, 1].size * p_sel))]

    bestSol = pop[sel_pool[0], :]

    for sol in sel_pool[1:len(sel_pool)]:
        if pop[sol, len(bestSol) - 1] < bestSol[len(bestSol) - 1]:
            bestSol = pop[sol, :]

    return bestSol


def swap_mutation(child, numberOfSwaps):
    """
    Performs swap mutation on a child solution by swapping pairs of genes.

    Parameters:
    -----------
    child : numpy.ndarray
        The solution (individual) on which mutation is performed.
    numberOfSwaps : int
        The number of times gene pairs should be swapped.

    Returns:
    --------
    child : numpy.ndarray
        The mutated solution with swapped genes.
    """

    for _ in range(numberOfSwaps):
        swapGenesPairs = np.random.choice(len(child), 2, replace=False)

        a, b = child[swapGenesPairs[0]], child[swapGenesPairs[1]]
        child[swapGenesPairs[0]], child[swapGenesPairs[1]] = b, a

    return child

def plotCheckBoard(sol):
    """
    Visualizes the given solution on a checkerboard pattern.

    Parameters:
    -----------
    sol : numpy.ndarray
        A 1D array representing the positions of queens on the chessboard.
        Each element indicates the column index of the queen in each row.

    Returns:
    --------
    None
    """

    def checkerboard(shape):
        """Creates a checkerboard pattern of the specified shape."""
        return np.indices(shape).sum(axis=0) % 2

    sol = sol - 1  
    size = len(sol)
    board = checkerboard((size, size)).astype('float64')
    
    for i in range(size):
        board[i, int(sol[i])] = 0.5  

    fig, ax = plt.subplots()
    ax.imshow(board, cmap=plt.cm.CMRmap, interpolation='nearest')
    plt.show()

# Parameters of the algorithm
generation = 150  # Number of generations
p_sel = 0.95      # Probability of Selection
p_m = 0.1         # Probability of Mutation
numberOfSwaps = 2 # Number of swaps during mutation
npop = 150        # Number of solutions   
size = 10         # Size of board and queens
ox_size = 2       # Variables changed during order crossover 


pop = np.zeros((npop, size), dtype=int)
for i in range(npop):
    pop[i, :] = np.random.permutation(size) + 1

fit = checkFitness(pop)
pop = np.hstack((pop, fit))

meanFit = np.zeros(generation)


for gen in range(generation):
    """
    Main loop for evolving the population over a specified number of generations.

    Parameters:
    -----------
    generation : int
        The total number of generations to run the evolutionary algorithm.

    In each generation, the following steps are performed:
    1. Select two parent solutions from the population.
    2. Generate offspring using order crossover from the selected parents.
    3. Apply mutation to the offspring based on a mutation probability.
    4. Evaluate the fitness of the new offspring.
    5. Combine the existing population with the new offspring.
    6. Sort the population by fitness and retain the best solutions.
    7. Calculate the mean fitness of the current population.
    """

    print(f"Generation: {gen + 1} / {generation}")

    parents = [selection(pop, p_sel), selection(pop, p_sel)]
    
    offsprings = order_crossover(parents[0][0:size], parents[1][0:size], ox_size)

    for child in range(len(offsprings)):
        r_m = round(random.random(), 2)
        if r_m <= p_m:
            offsprings[child] = swap_mutation(offsprings[child], numberOfSwaps)

    fitOff = checkFitness(offsprings)
    offsprings = np.hstack((offsprings, fitOff))

    pop = np.vstack([pop, offsprings])

    pop = pop[pop[:, size].argsort()][0:npop, :]

    meanFit[gen] = (pop[:, size]).mean()

bestSol = pop[np.argmin(pop[:, size]), :]

# Plot fitness evolution
plt.plot(meanFit)
plt.grid()
plt.title("Evolution of Fit (Mean)")
plt.ylabel("Fit Mean")
plt.xlabel("Generation")
plt.show()

# Output the best solution
print(f"Best Solution has {bestSol[size]} Conflict(s)")
plotCheckBoard(bestSol[0:size])


