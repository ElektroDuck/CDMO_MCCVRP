from z3 import *
import os
import time
import numpy as np
from utilities.utilities import compute_upper_bound
from utilities.utilities import compute_lower_bound
import argparse
import json


#Initialization of the instance
start = time.time()
parser = argparse.ArgumentParser()
parser.add_argument("inst")
args = parser.parse_args()
inst = int(args.inst)
instance = f"inst0{inst}.dat" if inst < 10 else f"inst{inst}.dat"


def create_result_dict(model, is_final):
    """
    This function, given a model, will create the result dict in JSON format
    """
    global start
    best_paths_dict = {}
    for i in range(n):
        for j in range(m+1):
            for k in range(m+1):
                if is_true(model.eval(paths[i][j][k])):
                    best_paths_dict[(j, k)] = i

    unordered_paths = [[] for i in range(n)]
    for k,v in best_paths_dict.items():
        unordered_paths[v].append(k)


    
    best_paths = []
    for path in unordered_paths:
        # Crea un dizionario per mappare partenza -> arrivo
        path_dict = {starting_point: ending_point for starting_point, ending_point in path}
        
        # Inizia dal nodo `m`
        ordered_paths = [(m, path_dict[m])]
        
        # Continua a cercare il prossimo nodo finché non torni a `m`
        while ordered_paths[-1][1] != m:
            partenza = ordered_paths[-1][1]
            arrivo = path_dict[partenza]
            ordered_paths.append((partenza, arrivo))
        
        best_paths.append(ordered_paths)


    # This line will generate a list of all the costs of the paths (unused)
    #costs = [sum(distances[path[0]][path[1]] for path in courier_path) for courier_path in best_paths]     
    best_paths = [[x[1] for x in path[:-1]] for path in best_paths]
    


    
    result_dict = {
    "time" : min(300,int(time.time() - start)),
    "optimal" : is_final,
    "obj" : int(model.eval(max_dist).as_string()),
    "sol" : best_paths
    }

    return result_dict


def callback(tmp_model):
    """
    This function will be used to save intermediate results
    """
    res_dict = create_result_dict(tmp_model, False)
    with open("tmp_output.json", "w") as f:
        json.dump(res_dict, f)




# Opening the instance
if os.path.exists("../Instances"):
    with open("../Instances/"+instance, 'r') as file:
        lines = file.readlines()
    
    num_vehicles = int(lines[0].strip())
    num_clients = int(lines[1].strip())
    
    vehicles_capacity = list(map(int, lines[2].strip().split()))
    packages_size = list(map(int, lines[3].strip().split()))

    distances = [list(map(int, line.strip().split())) for line in lines[4:]]


# parameter definition
n = num_vehicles
m = num_clients
ub = compute_upper_bound(distances)
lb = compute_lower_bound(distances, num_vehicles, num_clients)

# Define decision variables
paths = [[[Bool("courier[%i,%i,%i]" % (i, j, k)) for k in range(m+1)] for j in range(m+1)] for i in range(n)]
num_visit = [Int(f"num_visit{i}") for i in range(m)]


#  creating the solver
solver = Optimize()
# Setting the callback for intermediate results
solver.set_on_model(callback)

# Adding constraints

# Constraints on decision variables domains
for i in range(m):
    solver.add(And(num_visit[i] >= 0, num_visit[i] <= m-1))


# Each customer should be visited only once
for i in range(n):
    for k in range(m):
        solver.add(Implies(Sum([paths[i][j][k] for j in range(m+1)]) == 1, Sum([paths[i][k][j] for j in range(m+1)]) == 1))
        solver.add(And(Sum([paths[i][j][k] for i in range(n) for j in range(m + 1)]) == 1,
                        Sum([paths[i][k][j] for j in range(m + 1) for i in range(n)]) == 1))


# Subtour constraint
for i in range(n):
    for j in range(m):
        for k in range(m):
            solver.add(Implies(paths[i][j][k], num_visit[j] < num_visit[k]))

# Capacity constraint
for i in range(n):
    solver.add(Sum([paths[i][j][k]*packages_size[k] for k in range(m) for j in range(m+1)]) <= vehicles_capacity[i])


# paths[i][j][j] should be False for any i and any j Diagonal != 1
solver.add(Sum([paths[i][j][j] for j in range(m+1) for i in range(n)]) == 0)

# Each path should begin and end at the depot
for i in range(n):
    solver.add(And(Sum([paths[i][m][k] for k in range(m)]) == 1, Sum([paths[i][j][m] for j in range(m)]) == 1))

# Define the objective function and apply upper and lower bounds
max_dist = Int("max_dist")
solver.add(max_dist <= ub)
solver.add(max_dist >= lb)
for i in range(n):
    solver.add(Sum([paths[i][j][k]*distances[j][k] for j in range(m+1) for k in range(m+1)]) <= max_dist)


# Symmetry breaking constraint
for i1 in range(n):
    for i2 in range(n):
        if i1 < i2 and vehicles_capacity[i1] == vehicles_capacity[i2]:
            for j in range(m):
                for k in range(m):
                    solver.add(Implies(And(paths[i1][m][j], paths[i2][m][k]), j < k))


solver.minimize(max_dist)

# Run the solver
if solver.check() == sat:
    model = solver.model()
    result_dict = create_result_dict(model, True)
    with open("tmp_output.json", "w") as f:
        json.dump(result_dict, f)

# If the model is not sat...
else:
    result_dict = {
    "time" : 300,
    "optimal" : False,
    "obj" : 0,
    "sol" : []
    }
    with open("tmp_output.json", "w") as f:
        json.dump(result_dict, f)

