# Defined starting and goal states
initial_state = [ 1, 2, 0, 3 ]
goal_state = [ 1, 2, 3, 0 ]


# Displays the 1D board in 2D shape
def show_board(board, string):
    if board == None:
        return
    print(string)
    print(f'---------')
    print(f'| {board[0]} | {board[1]} |')
    print(f'---------')
    print(f'| {board[2]} | {board[3]} |')
    print(f'---------')
    print()


# Diplayed both the boards
show_board(initial_state, 'Initial State')
show_board(goal_state, 'Goal State')

# Displays the path from the initial state to the goal stat

def show_path(path):
    if path == None:
        print('No path found')
    else:
        print('Path:')
        for state in path:
            show_board(state, '')

# Depth-First Search algorithm
def dfs(state, goal_state, visited, path):
    if state == goal_state:
        path.append(state)
        return path
    visited.add(tuple(state))
    for i in range(len(state)):
        if state[i] == 0:
            row, col = divmod(i, 2)
            moves = []
            if row > 0:
                moves.append(i - 2)
            if row < 1:
                moves.append(i + 2)
            if col > 0:
                moves.append(i - 1)
            if col < 1:
                moves.append(i + 1)
            for move in moves:
                new_state = state[:]
                new_state[i], new_state[move] = new_state[move], new_state[i]
                if tuple(new_state) not in visited:
                    new_path = path[:]
                    new_path.append(state)
                    result = dfs(new_state, goal_state, visited, new_path)
                    if result != None:
                        return result
    return None

# Call the DFS algorithm with the initial state and an empty visited set and path
visited = set()
path = []
result = dfs(initial_state, goal_state, visited, path)

# Display the path from the initial state to the goal state
show_path(result)