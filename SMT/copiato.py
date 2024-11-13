from z3 import *
from timeit import default_timer as timer
import json
import numpy as np
from utilities.utilities import compute_upper_bound


def callback(tmp_model):
    print(f"\tObjective function value: {tmp_model.eval(max_dist)}")


def compute_bounds(distances, num_vehicles, num_clients):
    matrix_dist = np.array(distances) #transform in numpy matrix
    last_row = matrix_dist[-1, :]  # selects the last row
    last_column = matrix_dist[:, -1]  # selects the last column
    result = last_row + last_column
    low_bound = max(result)
    #up_bound = sum([max(matrix_dist[i] for i in range(num_vehicles-1, num_clients))])
    min_dist_bound = min(result)

    dist_sorted = matrix_dist[np.max(matrix_dist, axis=0).argsort()]
    up_bound = sum([max(dist_sorted[i]) for i in range(num_vehicles-1, num_clients+1)])

    return low_bound, min_dist_bound, up_bound

for i in range(1, 22):
    if i < 10:
        instance = "0"+str(i)
    else:
        instance = i
    with open(f"../Instances/inst{instance}.dat") as file:
        data = file.read().strip().splitlines()

    print("\nSolving instance: ",i)

    num_couriers = int(data[0])
    num_items = int(data[1])
    load_sizes = [int(i) for i in data[2].split()]
    item_sizes = [int(i) for i in data[3].split()]
    distances = []
    for i in range(num_items+1):
        row = [int(j) for j in data[4+i].split()]
        distances.append(row)

    ub,_ = compute_upper_bound(distances)
    lb,_,_ = compute_bounds(distances, num_couriers, num_items)
    # Define decision variables
    paths = [[[Bool("courier[%i,%i,%i]" % (i, j, k)) for k in range(num_items+1)] for j in range(num_items+1)] for i in range(num_couriers)]
    num_visit = [Int(f"num_visit{i}") for i in range(num_items)]

    solver = Optimize()
    solver.set_on_model(callback)
    callback.step_counter = 0
    solver.set("timeout", 10000) 




    max_dist = Int("max_dist")
    solver.add(max_dist <= ub)
    solver.add(max_dist >= lb)
    # Constraints on decision variables domains
    for i in range(num_items):
        solver.add(And(num_visit[i] >= 0, num_visit[i] <= num_items-1))

    # Each customer should be visited only once
    for i in range(num_couriers):
        for k in range(num_items):
            solver.add(Implies(Sum([paths[i][j][k] for j in range(num_items+1)]) == 1, Sum([paths[i][k][j] for j in range(num_items+1)]) == 1))
            solver.add(And(Sum([paths[i][j][k] for i in range(num_couriers) for j in range(num_items + 1)]) == 1,
                           Sum([paths[i][k][j] for j in range(num_items + 1) for i in range(num_couriers)]) == 1))

    # Subtour constraint
    for i in range(num_couriers):
        for j in range(num_items):
            for k in range(num_items):
                solver.add(Implies(paths[i][j][k], num_visit[j] < num_visit[k]))

    # Capacity constraint
    for i in range(num_couriers):
        solver.add(Sum([paths[i][j][k]*item_sizes[k] for k in range(num_items) for j in range(num_items+1)]) <= load_sizes[i])

    # paths[i][j][j] should be False for any i and any j
    solver.add(Sum([paths[i][j][j] for j in range(num_items+1) for i in range(num_couriers)]) == 0)

    # Each path should begin and end at the depot
    for i in range(num_couriers):
        solver.add(And(Sum([paths[i][num_items][k] for k in range(num_items)]) == 1, Sum([paths[i][j][num_items] for j in range(num_items)]) == 1))

    # Define the objective function
    for i in range(num_couriers):
        solver.add(Sum([paths[i][j][k]*distances[j][k] for j in range(num_items+1) for k in range(num_items+1)]) <= max_dist)

    # Symmetry breaking
    for i1 in range(num_couriers):
        for i2 in range(num_couriers):
            if i1 < i2 and load_sizes[i1] == load_sizes[i2]:
                for j in range(num_items):
                    for k in range(num_items):
                        solver.add(Implies(And(paths[i1][num_items][j], paths[i2][num_items][k]), j < k))

    solver.minimize(max_dist)
    try:
        start = timer()
        res = solver.check()
        end = timer()
        if res == sat:
            optimal = "true"
            print("The problem is satisfiable.", end=" ")
        elif res == unsat:
            optimal = "false"
            print("The problem is unsatisfiable.")
        else:
            optimal = "false"
            print("Unknown. It is impossible to determine whether the problem is satisfiable or not.")
    except Z3Exception as e:
        # Timeout
        print("Timeout occurred")
    model = solver.model()
    if res != unsat:
        best_paths_dict = {}
        for i in range(num_couriers):
            for j in range(num_items+1):
                for k in range(num_items+1):
                    if is_true(model.eval(paths[i][j][k])):
                        best_paths_dict[(i, j)] = k
        best_paths = [[] for i in range(num_couriers)]
        for i in range(num_couriers):
            k = num_items
            while k != num_items or len(best_paths[i]) == 0:
                if (i, k) in best_paths_dict.keys():
                    if best_paths_dict[(i, k)] != num_items:
                        best_paths[i].append(best_paths_dict[(i, k)]+1)
                    k = best_paths_dict[(i, k)]
                else:
                    break
    time = int(end-start)
    if time < 300 and optimal == "false":
        time = 300
    try:
        best_max_dist = model.eval(max_dist).as_long()
        print(best_max_dist)
        results = {
            "Z3": {
                "time": time,
                "optimal": optimal,
                "obj": best_max_dist,
                "sol": best_paths
            }
        }
    except AttributeError as e:
        print("No value for the objective function was found.")
        results = {}
    