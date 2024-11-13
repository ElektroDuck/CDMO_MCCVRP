import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.optimize import linear_sum_assignment

def compute_lower_bound(distances, num_vehicles, num_clients):
    matrix_dist = np.array(distances) #transform in numpy matrix
    last_row = matrix_dist[-1, :]  # selects the last row
    last_column = matrix_dist[:, -1]  # selects the last column
    result = last_row + last_column
    low_bound = max(result)
    
    return low_bound



def calculate_mccrp_bounds(distance_matrix):
    """
    Calculate approximate lower and upper bounds for MCCRP using distance matrix.
    
    Parameters:
    distance_matrix (numpy.ndarray): A symmetric matrix where element [i][j] is the distance between node i and node j.
    
    Returns:
    tuple: (lower_bound, upper_bound)
    """
    
    # 1. Lower Bound Calculation using Minimum Spanning Tree (MST)
    mst = minimum_spanning_tree(distance_matrix).toarray()
    
    # 2. Upper Bound Calculation using a Greedy Heuristic
    num_nodes = len(distance_matrix)
    visited = [False] * num_nodes
    total_cost = 0
    current_node = 0  # Start at an arbitrary node, e.g., node 0
    visited[current_node] = True
    
    for _ in range(num_nodes - 1):
        # Find the nearest unvisited node
        min_distance = float('inf')
        next_node = None
        
        for neighbor in range(num_nodes):
            if not visited[neighbor] and distance_matrix[current_node][neighbor] < min_distance:
                min_distance = distance_matrix[current_node][neighbor]
                next_node = neighbor
        
        # Move to the next closest node
        if next_node is not None:
            total_cost += min_distance
            visited[next_node] = True
            current_node = next_node
    
    # Return to the starting node to complete the tour
    upper_bound = total_cost + distance_matrix[current_node][0]
    
    return upper_bound




def compute_upper_bound(matrix):
    matrix = np.array(matrix)
    upper_bound = calculate_mccrp_bounds(matrix)
    return upper_bound