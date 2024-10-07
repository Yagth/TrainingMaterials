import random


def foo(x, y, z):
    return 6 * x**3 + 9 * y**2 + 90 * z - 25


def fitness(x, y, z):
    ans = foo(x, y, z)

    if ans == 0:
        return float("inf")

    return abs(1 / ans)


solutions = []

# Generate initial random solutions
for i in range(1000):
    solutions.append(
        (
            random.uniform(-5000, 5000),
            random.uniform(-5000, 5000),
            random.uniform(-5000, 5000),
        )
    )

for i in range(10000):
    rankedSolutions = []

    # Rank solutions to determine the best ones.
    for s in solutions:
        rankedSolutions.append((fitness(s[0], s[1], s[2]), s))

    rankedSolutions.sort()
    rankedSolutions.reverse()

    print(f"=== Gen {i} === Best solutions")
    print(rankedSolutions[0])

    # Terminate if good engough solution is found.
    if rankedSolutions[0][0] > 10000:
        break

    # Generate new solutions from the best 100 ones.
    # This involves crossing over best solutions and also mutating them.

    bestSolutions = rankedSolutions[:100]
    elements = []

    for s in bestSolutions:
        elements.append(s[1][0])
        elements.append(s[1][1])
        elements.append(s[1][2])

    newSolutions = []
    for _ in range(1000):
        a = random.choice(elements) * random.uniform(0.9, 1.1)
        b = random.choice(elements) * random.uniform(0.9, 1.3)
        c = random.choice(elements) * random.uniform(0.9, 1.5)

        newSolutions.append((a, b, c))

    solutions = newSolutions
