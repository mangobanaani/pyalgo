# Backtracking Algorithms

Collection of backtracking algorithms that solve constraint satisfaction problems by systematically searching through solution space.

## Algorithms Included

### N-Queens Problem (`n_queens.py`)
- **Problem**: Place N queens on N×N chessboard such that no two queens attack each other
- **Time Complexity**: O(N!) worst case, typically much better with pruning
- **Space Complexity**: O(N) for recursion stack and board representation
- **Description**: Classic constraint satisfaction problem using backtracking with constraint checking
- **Solutions**: Find one solution, count all solutions, find all unique solutions
- **Use Case**: Constraint satisfaction, algorithm design education

### Sudoku Solver (`sudoku_solver.py`)
- **Problem**: Fill 9×9 grid following Sudoku rules (each row, column, and 3×3 box contains digits 1-9)
- **Time Complexity**: O(9^(n×n)) worst case where n×n is empty cells
- **Space Complexity**: O(n×n) for recursion stack
- **Description**: Constraint-based solving with validity checking at each step
- **Features**: Solve any valid Sudoku puzzle, validate solutions
- **Use Case**: Puzzle solving, constraint propagation

### Knight's Tour (`knights_tour.py`)
- **Problem**: Find sequence of moves for knight to visit every square on chessboard exactly once
- **Time Complexity**: O(8^(n×n)) worst case
- **Space Complexity**: O(n×n) for board and recursion stack
- **Description**: Graph traversal problem using backtracking with move validation
- **Variants**: Open tour (end anywhere), closed tour (return to start)
- **Use Case**: Graph theory, Hamiltonian path problems

### Maze Solver (`maze_solver.py`)
- **Problem**: Find path from start to end position in maze avoiding obstacles
- **Time Complexity**: O(4^(n×m)) worst case for n×m maze
- **Space Complexity**: O(n×m) for visited tracking and recursion
- **Description**: Path-finding using backtracking with obstacle avoidance
- **Features**: Multiple path finding, shortest path options
- **Use Case**: Robotics, game AI, path planning

### Crossword Solver (`crossword_solver.py`)
- **Problem**: Fill crossword grid with words from dictionary satisfying intersection constraints
- **Time Complexity**: Exponential in number of words and grid size
- **Space Complexity**: O(grid size + dictionary size)
- **Description**: Constraint satisfaction with word placement and intersection validation
- **Features**: Dictionary-based word fitting, constraint propagation
- **Use Case**: Puzzle generation, natural language processing

## Usage Examples

### N-Queens Problem
```python
from backtracking.n_queens import solve_n_queens, count_n_queens_solutions

# Solve 8-Queens problem
n = 8
solution = solve_n_queens(n)
if solution:
    print(f"Solution for {n}-Queens:")
    for row in solution:
        print(['Q' if col == row else '.' for col in range(n)])
else:
    print(f"No solution found for {n}-Queens")

# Count all solutions
count = count_n_queens_solutions(n)
print(f"Total solutions for {n}-Queens: {count}")

# Solve for different board sizes
for size in range(4, 9):
    solutions = count_n_queens_solutions(size)
    print(f"{size}-Queens has {solutions} solutions")
```

### Sudoku Solver
```python
from backtracking.sudoku_solver import solve_sudoku, is_valid_sudoku

# Example Sudoku puzzle (0 represents empty cells)
puzzle = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
]

# Solve the puzzle
if solve_sudoku(puzzle):
    print("Solved Sudoku:")
    for row in puzzle:
        print(row)
else:
    print("No solution exists for this puzzle")

# Validate a completed puzzle
if is_valid_sudoku(puzzle):
    print("Solution is valid!")
```

### Knight's Tour
```python
from backtracking.knights_tour import knights_tour, is_valid_move

# Solve Knight's Tour for 8x8 board
board_size = 8
start_x, start_y = 0, 0

solution = knights_tour(board_size, start_x, start_y)
if solution:
    print(f"Knight's Tour solution ({board_size}x{board_size}):")
    for row in solution:
        print([f"{num:2d}" for num in row])
else:
    print("No solution found")

# Try different board sizes
for size in [5, 6, 7, 8]:
    tour = knights_tour(size, 0, 0)
    if tour:
        print(f"{size}x{size} board: Solution found")
    else:
        print(f"{size}x{size} board: No solution")
```

### Maze Solver
```python
from backtracking.maze_solver import solve_maze, print_maze_solution

# Define maze (1 = wall, 0 = path)
maze = [
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0]
]

start = (0, 0)
end = (4, 4)

# Find path through maze
path = solve_maze(maze, start, end)
if path:
    print("Maze solution found!")
    print_maze_solution(maze, path)
    print(f"Path: {path}")
else:
    print("No path found through maze")

# Create and solve random maze
from backtracking.maze_solver import generate_random_maze
random_maze = generate_random_maze(10, 10, obstacle_probability=0.3)
path = solve_maze(random_maze, (0, 0), (9, 9))
```

### Crossword Solver
```python
from backtracking.crossword_solver import solve_crossword, create_crossword_grid

# Define crossword structure and word list
grid_template = [
    ['_', '_', '_', '#', '_', '_', '_'],
    ['_', '#', '_', '#', '_', '#', '_'],
    ['_', '_', '_', '_', '_', '_', '_'],
    ['#', '_', '#', '_', '#', '_', '#'],
    ['_', '_', '_', '_', '_', '_', '_']
]

word_list = ["CAT", "DOG", "RAT", "BAT", "HAT", "MAT", "FAT"]

# Solve crossword
solution = solve_crossword(grid_template, word_list)
if solution:
    print("Crossword solution:")
    for row in solution:
        print(' '.join(row))
else:
    print("No solution found for crossword")
```

## Algorithm Analysis

### Time Complexity Comparison

| Problem | Best Case | Average Case | Worst Case | Typical Performance |
|---------|-----------|--------------|------------|-------------------|
| N-Queens | O(N!) | O(N!) | O(N!) | Much better with pruning |
| Sudoku | O(1) | O(9^k) | O(9^81) | Fast for valid puzzles |
| Knight's Tour | O(8^k) | O(8^(N²)) | O(8^(N²)) | Varies by heuristics |
| Maze Solving | O(1) | O(4^(N×M)) | O(4^(N×M)) | Good with pruning |
| Crossword | O(W!) | Exponential | Exponential | Depends on constraints |

*Where N = board size, k = empty cells, W = number of words*

### Space Complexity

| Problem | Recursion Stack | Additional Space | Total Space |
|---------|----------------|------------------|-------------|
| N-Queens | O(N) | O(N) | O(N) |
| Sudoku | O(81) | O(1) | O(1) |
| Knight's Tour | O(N²) | O(N²) | O(N²) |
| Maze Solving | O(N×M) | O(N×M) | O(N×M) |
| Crossword | O(W) | O(grid) | O(grid + W) |

## Optimization Techniques

### Constraint Propagation
- **Early Pruning**: Eliminate invalid choices as soon as detected
- **Forward Checking**: Check constraints before making moves
- **Arc Consistency**: Maintain consistency between related variables

### Heuristics
- **Most Constrained Variable**: Choose variable with fewest legal values
- **Least Constraining Value**: Choose value that eliminates fewest options
- **Warnsdorff's Heuristic**: For Knight's Tour, move to square with fewest onward moves

### Implementation Optimizations
- **Bit Manipulation**: Use bitsets for faster constraint checking
- **Memoization**: Cache partial solutions where applicable
- **Iterative Deepening**: For bounded search spaces

## Common Patterns

### Backtracking Template
```python
def backtrack(state, solution):
    if is_complete(state):
        return solution
    
    for choice in get_choices(state):
        if is_valid(choice, state):
            make_choice(choice, state)
            result = backtrack(state, solution)
            if result is not None:
                return result
            undo_choice(choice, state)  # Backtrack
    
    return None
```

### Constraint Checking
- **Local Constraints**: Check immediate conflicts
- **Global Constraints**: Verify overall solution validity
- **Incremental Checking**: Update constraints as choices are made

## Applications

### Educational
- **Algorithm Learning**: Understanding recursive problem solving
- **Constraint Satisfaction**: Learning systematic search techniques
- **Complexity Analysis**: Studying exponential algorithms

### Practical Applications
- **Puzzle Solving**: Sudoku, crosswords, logic puzzles
- **Game AI**: Move generation and evaluation
- **Scheduling**: Resource allocation with constraints
- **Configuration**: System setup with dependencies

### Research Applications
- **Combinatorial Optimization**: Finding optimal solutions
- **Artificial Intelligence**: Constraint satisfaction problems
- **Operations Research**: Resource planning and allocation

## When to Use Backtracking

### Good Fit:
- Problem has discrete choice points
- Constraints can be checked incrementally
- Solution space can be pruned effectively
- Complete search is feasible or required

### Poor Fit:
- Continuous optimization problems
- Very large solution spaces without good pruning
- Problems with better specialized algorithms
- Real-time applications requiring guaranteed response time

## Advanced Considerations

### Parallelization
- **Work Stealing**: Distribute search tree exploration
- **Branch and Bound**: Parallel evaluation of solution bounds
- **GPU Acceleration**: Parallel constraint checking

### Alternative Approaches
- **Dynamic Programming**: For overlapping subproblems
- **Greedy Algorithms**: For approximation solutions
- **Constraint Programming**: Specialized CP solvers
- **Metaheuristics**: Genetic algorithms, simulated annealing
