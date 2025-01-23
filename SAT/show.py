from convertion_functions import *

def showMCCVRP(orders, distances_bin, obj_value, assignments):
    """Function to display a found solution of the Multiple Couriers Planning problem

    Args:
        orders (list[list[bool]]): matrix representing the order of delivery of each object 
                                   in its route, namely orders[j][k] == True iff object j is delivered as k-th by its courier 
        distances_bin (list[list[bool]]): for each courier, its travelled distance represented in binary
        obj_value (int): the objective value obtained
        assignments (list[list[bool]]): matrix of assignments, assignments[i][j] = True iff courier i delivers
                                        object j, false otherwise.
    """
    distances = [bin_to_int(d) for d in distances_bin]

    print(f"Objective value: {obj_value}")
    print(f"Solution found: ")
    m = len(assignments)
    n = len(assignments[0])
    routes = [[0 for j in range(n)] for i in range(m)]
    for node in range(n):
        for time in range(n):
            if orders[node][time]:
                for courier in range(m):
                    if assignments[courier][node]:
                        routes[courier][time] = node+1
                        break
                break

    routes = [[x for x in row if x != 0] for row in routes] # remove trailing zeros
    for courier in range(m):
        print("Depo -> " +
              ' -> '.join([str(node) for node in routes[courier]]) +
              f' -> Depo: distance {distances[courier]}')