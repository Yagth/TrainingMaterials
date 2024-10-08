from functools import lru_cache

# Example graph represented as a distance matrix
graph = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
]
n = len(graph)

# Parent table to track the path
parent = [[-1] * n for _ in range(1 << n)]  # To store the parent of each node


@lru_cache(None)
def tsp(mask, pos):
    # Base case: if all cities are visited
    if mask == (1 << n) - 1:
        return graph[pos][0]  # Return cost to go back to the starting city

    # Try all unvisited cities and calculate the minimum cost
    ans = float('inf')
    for city in range(n):
        if not (mask & (1 << city)):  # If the city is not visited
            new_mask = mask | (1 << city)
            new_cost = graph[pos][city] + tsp(new_mask, city)
            if new_cost < ans:
                ans = new_cost
                parent[mask][pos] = city  # Track the best next city

    return ans

# Function to reconstruct the optimal path


def get_optimal_path():
    mask = 1  # Start with only the first city visited
    pos = 0  # Starting position at city 0
    path = [0]  # Start the path with city 0

    while mask != (1 << n) - 1:  # Until all cities are visited
        next_city = parent[mask][pos]
        path.append(next_city)
        mask |= (1 << next_city)
        pos = next_city

    path.append(0)  # Return to the starting city
    return path


# Start the journey from the first city with only the first city visited
initial_mask = 1  # Only the first city visited
min_cost = tsp(initial_mask, 0)

# Get the optimal path
optimal_path = get_optimal_path()

print("The minimum cost to visit all cities and return is:", min_cost)
print("The optimal path is:", optimal_path)
