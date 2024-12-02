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

instance_data = json.loads(str(args.inst).replace("'","\""))



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


num_vehicles = instance_data["num_vehicles"]
num_clients = instance_data["num_clients"]
vehicles_capacity = instance_data["vehicles_capacity"]
packages_size = instance_data["packages_size"]
distances = instance_data["distances"]

# parameter definition
n = num_vehicles
m = num_clients
ub = compute_upper_bound(distances)
lb = compute_lower_bound(distances, num_vehicles, num_clients)

# Define decision variables
#paths = [[[Bool("courier[%i,%i,%i]" % (i, j, k)) for k in range(m+1)] for j in range(m+1)] for i in range(n)] #OLD
paths = [[[Bool("courier[%i,%i,%i]" % (i, j, k)) for k in range(n)] for j in range(m+1)] for i in range(m+1)]
num_visit = [Int(f"num_visit{i}") for i in range(m)]

#  creating the solver
solver = Optimize()
# Setting the callback for intermediate results
solver.set_on_model(callback)

# Adding constraints

# Constraints on decision variables domains
for i in range(m):
    solver.add(And(num_visit[i] >= 0, num_visit[i] <= m-1))

# Constraint I: Da capire che cazzo è
for k in range(n):
    for j in range(m):
        solver.add(Implies(Sum([paths[i][j][k] for i in range(m+1)]) == 1, Sum([paths[j][i][k] for i in range(m+1)]) == 1))

# Constraint II: Each destination is reached by a courier departing from a sound origin (plus, each client is visited by only one courier)
for j in range(m):
    solver.add(And(Sum([paths[i][j][k] for i in range(m+1) for k in range(n)]) == 1,
                   Sum([paths[j][i][k] for i in range(m+1) for k in range(n)]) == 1))


# Constraint III: Subtour constraint
for k in range(n):
    for i in range(m):
        for j in range(m):
            solver.add(Implies(paths[i][j][k], num_visit[i] < num_visit[j]))



# Define the objective function and apply upper and lower bounds
max_dist = Int("max_dist")
solver.add(max_dist <= ub)
solver.add(max_dist >= lb)



# Constraint IV: Capacity constraint
# Constraint V: diagonal = 0
# Constraint VI: Start and End at depot
# Constraint VII: Obj function
for k in range(n):
    solver.add(Sum([paths[i][j][k] * packages_size[j] for i in range(m+1) for j in range(m)]) <= vehicles_capacity[k])
    solver.add(Sum([paths[i][i][k] for i in range(m+1)]) == 0)
    solver.add(And(Sum([paths[i][m][k] for i in range(m)]) == 1, Sum([paths[m][j][k] for j in range(m)]) == 1))
    #Forse spostare in un altro ciclo
    solver.add(Sum([paths[i][j][k] * distances[i][j] for i in range(m+1) for j in range(m+1)]) <= max_dist)



# Constraint VIII: Sym breaking

for i1 in range(n):
    for i2 in range(n):
        if i1 < i2 and vehicles_capacity[i1] == vehicles_capacity[i2]:
            for i in range(m):
                for j in range(m):
                    solver.add(Implies(And(paths[m][j][i1], paths[m][k][i2]), i < j))


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

