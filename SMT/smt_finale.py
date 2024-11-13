from z3 import *
import os
import time
import numpy as np
from utilities.utilities import compute_upper_bound
from utilities.utilities import compute_lower_bound



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

    ub = compute_upper_bound(distances)
    lb = compute_lower_bound(distances, num_vehicles, num_clients)


    # Define decision variables
    paths = [[[Bool("courier[%i,%i,%i]" % (i, j, k)) for k in range(m+1)] for j in range(m+1)] for i in range(n)]
    num_visit = [Int(f"num_visit{i}") for i in range(m)]


    # Creo un solver Z3
    solver = Optimize()

    # Aggiungo i vincoli:
    
    # Constraints on decision variables domains
    for i in range(m):
        solver.add(And(num_visit[i] >= 0, num_visit[i] <= m-1))

    # paths[i][j][j] should be False for any i and any j Diagonal != 1
    solver.add(Sum([paths[i][j][j] for j in range(m+1) for i in range(n)]) == 0)
    
    # Each path should begin and end at the depot
    for i in range(n):
        solver.add(And(Sum([paths[i][m][k] for k in range(m)]) == 1, Sum([paths[i][j][m] for j in range(m)]) == 1))
    
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

    
    
    
    max_dist = Int("max_dist")
    solver.add(max_dist <= ub)
    solver.add(max_dist >= lb)
    # Define the objective function
    for i in range(n):
        solver.add(Sum([paths[i][j][k]*distances[j][k] for j in range(m+1) for k in range(m+1)]) <= max_dist)
    

    # Symmetry breaking (non so come funzioni)
    for i1 in range(n):
        for i2 in range(n):
            if i1 < i2 and vehicles_capacity[i1] == vehicles_capacity[i2]:
                for j in range(m):
                    for k in range(m):
                        solver.add(Implies(And(paths[i1][m][j], paths[i2][m][k]), j < k))

    
    solver.minimize(max_dist)
    
    if solver.check() == sat:
        model = solver.model()
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
        for percorso in unordered_paths:
            # Crea un dizionario per mappare partenza -> arrivo
            percorso_dict = {partenza: arrivo for partenza, arrivo in percorso}
            
            # Inizia dal nodo `m`
            percorso_ordinato = [(m, percorso_dict[m])]
            
            # Continua a cercare il prossimo nodo finché non torni a `m`
            while percorso_ordinato[-1][1] != m:
                partenza = percorso_ordinato[-1][1]
                arrivo = percorso_dict[partenza]
                percorso_ordinato.append((partenza, arrivo))
            
            best_paths.append(percorso_ordinato)


        costs = [sum(distances[path[0]][path[1]] for path in courier_path) for courier_path in best_paths]        
        best_paths = [[x[1] for x in path[:-1]] for path in best_paths]
        




        return True, model[max_dist], best_paths,costs
    else:
        print("Nessuna soluzione trovata.")
        return False, None




for i in range(1,22):
    instance = f"inst0{i}.dat" if i < 10 else f"inst{i}.dat"
    start = time.time()
    result, max_dist, paths,costs = solve(instance)
    end = time.time()
    if result:
        print(f"Instance {i} solved in {round(end-start, 2)}s with a max_dist of {max_dist}")
        for ix, path in enumerate(paths):
            print(f"\t path {ix}: {path} - cost: {costs[ix]}")
    else:
        print(f"instance {i} not solved (took {round(end-start, 2)}s)")

    result_dict = {
        "time" : min(round(end-start, 2), 300),
        "optimal" : result,
        "obj" : max_dist,
        "sol" : paths 
    }
