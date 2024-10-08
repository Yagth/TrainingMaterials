def is_valid(queen_positions, row, col):
    """
    Check if placing a queen at (row, col) is valid.
    :param queen_positions: Current positions of queens.
    :param row: Current row to place the queen.
    :param col: Column to place the queen.
    :return: True if the position is valid, False otherwise.
    """
    for r in range(row):
        c = queen_positions[r]
        # Check for column and diagonal attacks
        if c == col or abs(c - col) == abs(r - row):
            return False
    return True


def backtrack(n, row, queen_positions, solutions):
    """
    Recursive backtracking function to find all solutions.
    :param n: Size of the board (n x n).
    :param row: Current row to place the queen.
    :param queen_positions: List of queen positions.
    :param solutions: List to store valid solutions.
    """
    if row == n:
        # All queens are placed successfully
        solutions.append(queen_positions[:])  # Append a copy of the solution
        return

    for col in range(n):
        if is_valid(queen_positions, row, col):
            # Place queen
            queen_positions[row] = col
            backtrack(n, row + 1, queen_positions, solutions)
            # Backtrack (remove the queen is implicit since we overwrite it in the next iteration)


def n_queens_backtracking(n):
    """
    Solves the N-Queens problem using backtracking.
    :param n: The size of the chessboard (n x n) and the number of queens.
    :return: A list of solutions, where each solution is a valid arrangement of queens.
    """
    solutions = []
    queen_positions = [-1] * n  # Initialize positions of queens
    backtrack(n, 0, queen_positions, solutions)
    return solutions


# Example usage for 4-Queens:
n = 4
solutions = n_queens_backtracking(n)

# Print all solutions:
for sol in solutions:
    print(sol)
