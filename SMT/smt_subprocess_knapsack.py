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

# Get the instance to read
instance_number = int(args.inst)




#Read the instance utility function
def extract_data_from_dat(instance_number):
    instance_number = str(instance_number) if instance_number >= 10 else "0"+str(instance_number)
    instance_path = f"{os.getcwd()}/Instances/inst{instance_number}.dat"
    with open(instance_path, 'r') as file:
        lines = file.readlines()
    
    num_vehicles = int(lines[0].strip())
    num_clients = int(lines[1].strip())
    
    vehicles_capacity = list(map(int, lines[2].strip().split()))
    packages_size = list(map(int, lines[3].strip().split()))

    distances = [list(map(int, line.strip().split())) for line in lines[4:]]

    return num_vehicles, num_clients, vehicles_capacity, packages_size, distances


def create_result_dict(model, is_final):
    """
    This function, given a model, will create the result dict in JSON format
    """
    global start
    best_paths_dict = {}
    for i in range(m+1):
        for j in range(m+1):
            for k in range(n):
                if is_true(model.eval(paths[i][j][k])):
                    best_paths_dict[(i, j)] = k

    unordered_paths = [[] for _ in range(n)]
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

# Read the instance
num_vehicles, num_clients, vehicles_capacity, packages_size, distances = extract_data_from_dat(instance_number)

# parameter definition
n = num_vehicles
m = num_clients
ub = compute_upper_bound(distances)
lb = compute_lower_bound(distances, num_vehicles, num_clients)

# Define decision variables
paths = [[[Bool("courier[%i,%i,%i]" % (i, j, k)) for k in range(n)] for j in range(m+1)] for i in range(m+1)]
num_visit = [Int(f"num_visit{i}") for i in range(m)]
y = [[Bool("has[%i,%i]" % (i, k)) for k in range(n)] for i in range(m)]

#  creating the solver
solver = Optimize()
# Setting the callback for intermediate results
solver.set_on_model(callback)

# Adding constraints
# Constraint I: limit on decision variables domains
for i in range(m):
    solver.add(And(num_visit[i] >= 0, num_visit[i] <= m-1))



for j in range(m):
    for k in range(n):
        # Constraint II: Coherence (If i reach a client, i then must depart from that client)
        solver.add(Sum([paths[i][j][k] for i in range(m+1)]) == Sum([paths[j][i][k] for i in range(m+1)]))


# Constraint III: Subtour constraint
for k in range(n):
    for i in range(m):
        for j in range(m):
            solver.add(Implies(paths[i][j][k], num_visit[i] < num_visit[j]))



max_dist = Int("max_dist")

# Constraint IV: diagonal = 0
# Constraint V: Start and End at depot
# Constraint VI: Obj function
for k in range(n):
    solver.add(Sum([paths[i][i][k] for i in range(m+1)]) == 0)
    solver.add(And(Sum([paths[i][m][k] for i in range(m)]) == 1, Sum([paths[m][j][k] for j in range(m)]) == 1))
    solver.add(Sum([paths[i][j][k] * distances[i][j] for i in range(m+1) for j in range(m+1)]) <= max_dist)

# Constraint VII: Channelling
for k in range(n):
    for j in range(m):
        solver.add(Sum([paths[i][j][k] for i in range(m+1)]) == y[j][k]) # Modified

# Constraint VIII: Capacity constraint
for k in range(n):
    solver.add(Sum([y[i][k] * packages_size[i] for i in range(m)]) <= vehicles_capacity[k]) # Modified

# Constraint IX: Uniqueness
for j in range(m):
    solver.add(Sum([y[j][k] for k in range(n)]) == 1) # Modified

# Define the objective function and apply upper and lower bounds
solver.add(max_dist <= ub)
solver.add(max_dist >= lb)




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

