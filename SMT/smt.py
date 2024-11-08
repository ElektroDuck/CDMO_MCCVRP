from z3 import *
import os
import time
import numpy as np
from upper_bound import compute_upper_bound
import sys


def callback(tmp_model):
    try:
        #print(f"\tIntermediate objective function value: {tmp_model.eval(max_dist)}")
        pass
    except:
        pass

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

def solve(instance):
    if os.path.exists("../Instances"):
        with open("../Instances/"+instance, 'r') as file:
            lines = file.readlines()
        
        num_vehicles = int(lines[0].strip())
        num_clients = int(lines[1].strip())
        
        vehicles_capacity = list(map(int, lines[2].strip().split()))
        packages_size = list(map(int, lines[3].strip().split()))

        distances = [list(map(int, line.strip().split())) for line in lines[4:]]

    # Definisco i parametri
    n = num_vehicles  # numero di righe
    m = num_clients  # numero di colonne
    ub,_ = compute_upper_bound(distances)
    lb,_,_ = compute_bounds(distances, num_vehicles, num_clients)

    # Define decision variables
    paths = [[[Bool("courier[%i,%i,%i]" % (i, j, k)) for k in range(m+1)] for j in range(m+1)] for i in range(n)]
    num_visit = [Int(f"num_visit{i}") for i in range(m)]
    

    # Creo un solver Z3
    solver = Optimize()
    solver.set_on_model(callback)
    solver.set("timeout", 1)



    # Aggiungo i vincoli:

    max_dist = Int("max_dist")
    solver.add(max_dist <= ub)
    solver.add(max_dist >= lb)
    
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
    
    # Symmetry breaking (non so come funzioni)
    for i1 in range(n):
        for i2 in range(n):
            if i1 < i2 and vehicles_capacity[i1] == vehicles_capacity[i2]:
                for j in range(m):
                    for k in range(m):
                        solver.add(Implies(And(paths[i1][m][j], paths[i2][m][k]), j < k))

                        
    # Define the objective function
    for i in range(n):
        solver.add(Sum([paths[i][j][k]*distances[j][k] for j in range(m+1) for k in range(m+1)]) <= max_dist)
    


    # Ottimizzo
    solver.minimize(max_dist)

    try:
        res = solver.check()
        if res == sat:
            model = solver.model()
            #print("Distanza massima minimizzata:", model[max_distance])
            #print(f"Distanza ottimizzata : ", model[max_dist])
            #print("\n")
            return True, model[max_dist]
        else:
            print("Nessuna soluzione trovata.")
            return False, None
    except Z3Exception as e:
        print("Timeout!")
        return False, None



for i in range(1,22):
    print("")
    instance = f"inst0{i}.dat" if i < 10 else f"inst{i}.dat"
    print("Solving instance ", instance)
    start = time.time()
    result, max_dist = solve(instance)
    end = time.time()
    if result:
        print(f"\tInstance {i} solved in {round(end-start, 2)}s with a max_dist of {max_dist}")
    else:
        print(f"\tinstance {i} not solved (took {round(end-start, 2)}s)")
