def check_all_hamiltonian(tensor):
    """Function to check that all the paths represented in tensor are hamiltonian cycles

    Args:
        tensor (list[list[list[bool]]]): list of adjacency matrices over 2-regular graphs

    Returns:
        bool: true iff all paths in tensor are hamiltonian cycles, false otherwise
    """
    m = len(tensor)
    for i in range(m):
        if not check_hamiltonian(tensor[i]):
            return False
    return True


def check_hamiltonian(matrix):
    """Function to check that the given adjancency matrix over 2-regular graph is a hamiltonian cycle, i.e. the graph is connected

    Args:
        matrix (list[list[bool]]): adjacency matrix over 2-regular graph

    Returns:
        bool: true iff the given adjancency matrix is a hamiltonian cycle, false otherwise
    """
    n = len(matrix)
    visited = [0] * n
    v = n - 1
    while visited[v] == 0:
        visited[v] = 1
        for k in range(n):
            if matrix[v][k] == True:
                v = k
                break
    num_vertices = sum(row.count(True) for row in matrix)
    return (sum(visited) == num_vertices)