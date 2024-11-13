import gurobipy as gp
from gurobipy import GRB, Model, quicksum
import os, json, math, time
from upper_bound import compute_upper_bound
import numpy as np


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


for i in range(1,22):
    if i<10:
        instance=f"0{i}"
    else:
        instance=f"{i}"
    #file_name = f"../../Instances/inst{instance}.dat"

    #convert the file name to windows format
    file_name = os.path.join('Instances', f'inst{instance}.dat')
    

    """LICENSE"""
    # LICENSE FOR ACADEMIC VERSION OF GUROBI
    # Create an environment with your WLS license

    params = {
    "WLSACCESSID": '216dd889-fa22-449b-a888-e35218548ac7',
    "WLSSECRET": 'c4660e0b-df0a-49a3-b518-0791ddc21565',
    "LICENSEID": 2581103,
    }
    env = gp.Env(params=params)


    '''Reading from the file and visualization'''
    f = open(file_name, "r")

    num_couriers = int(f.readline())
    num_items = int(f.readline())
    load_size = [int(x) for x in f.readline().split()]
    item_size = [int(x) for x in f.readline().split()]
    distances = []
    for i in range(num_items+1):
        distances.append([int(x) for x in f.readline().split()])
    f.close()

    ub,_ = compute_upper_bound(distances)
    lb,_,_ = compute_bounds(distances, num_couriers, num_items)

    num_customers = num_items
    CUSTOMERS = list(range(1,num_customers+1))
    V = [0] + CUSTOMERS
    m = num_couriers
    COURIERS = list(range(1,m+1))

    model = Model("VRP",env=env)
    x = model.addVars(V,V,COURIERS, vtype=GRB.BINARY, name="x_ijk")
    y = model.addVars(V,COURIERS, vtype=GRB.BINARY, name="y_ik")
    u = model.addVars(CUSTOMERS, COURIERS, vtype=GRB.CONTINUOUS, name="u_ik")
    max_distance = model.addVar(name='max_distance')
    obj_fn = quicksum(distances[i-1][j-1]*x[i,j,k] for i in V for j in V for k in COURIERS)
    model.setObjective(max_distance, sense=GRB.MINIMIZE)
    for k in COURIERS:
        model.addConstr(quicksum(x[i,j,k]*distances[i-1][j-1] for i in V for j in V) <= max_distance)

    '''CONSTRAINTS'''
    model.addConstr(max_distance <= ub)
    model.addConstr(max_distance >= lb)

    for i in CUSTOMERS:
        model.addConstr(quicksum(y[i,k] for k in COURIERS)==1)
        model.addConstr(quicksum(x[i,j,k] for j in V for k in COURIERS)==1)
        
    for j in CUSTOMERS:
        model.addConstr(quicksum(x[i,j,k] for i in V for k in COURIERS)==1)

    model.addConstr(quicksum(y[0,k] for k in COURIERS)==m)

    for k in COURIERS:
        model.addConstr(quicksum(y[i,k]*item_size[i-1] for i in CUSTOMERS)<=load_size[k-1])
        model.addConstr(quicksum(x[j,j,k] for j in V)==0)
        model.addConstr(quicksum(x[i,0,k] for i in V)==1)

    for i in CUSTOMERS:
        for k in COURIERS:
            model.addConstr(quicksum(x[i,j,k] for j in V)==quicksum(x[j,i,k] for j in V))
            model.addConstr(quicksum(x[j,i,k] for j in V)==y[i,k])
            
    for k in COURIERS:
        for j in V:
            model.addConstr(quicksum(x[i,j,k] for i in V)==quicksum(x[i,j,k] for i in V))
    
    for k in COURIERS:
        for i in CUSTOMERS:
            for j in CUSTOMERS:
                if i != j:
                    model.addConstr(u[i, k] - u[j, k] + num_customers * x[i, j, k] <= num_customers - 1)

    for i in CUSTOMERS:
        for j in CUSTOMERS:
            if i != j:
                model.addConstr(u[i,k] - u[j,k] + num_customers * x[i, j, k] <= num_customers - 1)

    time_limit = 300
    model.setParam(GRB.Param.TimeLimit, time_limit)
    
    start_time = time.time()
    try:
        model.optimize()
    except Exception as e:
        print(f"Optimization terminated due to time limit ({time_limit} seconds).")
        
    elapsed_time = time.time() - start_time 

    # Check the optimization status and retrieve the solution if available
    if model.status == GRB.OPTIMAL:
        print("Optimal solution found!")
        print("Optimal Objective Value:", model.objVal)
        for v in model.getVars():
            print(v.varName, v.x)
    elif model.status == GRB.TIME_LIMIT:
        print(f"Optimization reached the time limit ({time_limit} seconds).")
    else:
        print("Optimization did not converge to an optimal solution.")
        
        
    def make_json(filename, solvers, times, distances, solutions, is_optimal_vec):
        solv = {}
        for solver, max_dist_sol, solution, time, is_optimal in zip(solvers, distances, solutions, times, is_optimal_vec):
            data = {}
            print(is_optimal)
            data['time'] = int(time)
            data['optimal'] = is_optimal
            data['obj'] = max_dist_sol
            data['sol'] = solution
            solv[solver] = data
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as outfile:
            json.dump(solv, outfile)
            print(f"\nJSON file {filename} created successfully")
            
    if model.ObjVal == math.inf:
        json_file_path = f"res/MIP/inst{instance}.json"
        opt = False
        elapsed_time=300
    
        # Create the file if it doesn't exist
        if not os.path.exists(json_file_path):
            make_json(json_file_path, ["Gurobi"], [elapsed_time], [model.objVal], [], [opt])

        # Read the file from the same path where it was written
        with open(json_file_path) as json_file:
            data = json.load(json_file)
            print(data)
            make_json(json_file_path, ["Gurobi"], [elapsed_time], [model.objVal], [], [opt])
            print(data)
    else:
        def retrieve_elements(middle,first):
            for i in middle:
                if i[0]==first:
                    return i
                
        def compute_routes(x):
            routes = {}
            for (i, j, k) in x:
                if x[(i, j, k)].x == 1:
                    if k not in routes:
                        routes[k] = [(i, j)]
                    else:
                        routes[k].append((i, j))
            #reordering
            for k in routes:
                start = next((t for t in routes[k] if t[0] == 0), None)
                end = next((t for t in routes[k] if t[1] == 0), None)
                if start and end:
                    routes[k].remove(start)
                    routes[k].remove(end)
                    middle=[t for t in routes[k] if t != start and t != end]
                    sorted = []
                    token = start[1]
                    for el in routes[k]:
                        element = retrieve_elements(middle,token)
                        sorted.append(element)
                        token = element[1]
                    routes[k] = [start] + sorted + [end]
            return routes
        
        routes = compute_routes(x)
        for k, route in sorted(routes.items(), key=lambda item: item[0]):
            print(f"Courier {k} route: {route}")

        sol=[]
        for k in sorted(routes.keys()):
            s=[]
            for i in range(1,len(routes[k])):
                s.append(routes[k][i][0])
            sol.append(s)
        print("solution vector:",sol, end="\n\n")
        
        for k in routes:
            dist=0
            for i in range(0,len(routes[k])):
                dist+=distances[routes[k][i][0]-1][routes[k][i][1]-1]
            print(f"Courier {k} distance: {dist}")

        def compute_items_carried(x, num_items):
            items_carried = {}
            for (i, j, k) in x:
                if x[(i, j, k)].x == 1:
                    if k not in items_carried:
                        items_carried[k] = [i]
                    elif i != num_items+1:
                        items_carried[k].append(i)
            return items_carried

        items_carried = compute_items_carried(x, num_items)
        for k, items in items_carried.items():
            items.remove(0)
            print(f"Courier {k} carries items: {items}")

        print("\n")
        for k,items in items_carried.items():
            load=0
            for i in items_carried[k]:
                load+=item_size[i-1]
            print(f"Courier {k} load: {load} -> max load: {load_size[k-1]}")

        def compute_total_distance(x, distances):
            total_distance = {}
            for (i, j, k) in x:
                if x[(i, j, k)].X == 1:
                    if k not in total_distance:
                        total_distance[k] = distances[i-1][j-1]
                    else:
                        total_distance[k] += distances[i-1][j-1]
            return total_distance
        total_distance = compute_total_distance(x, distances)
        for k, distance in total_distance.items():
            print(f"Total distance traveled by vehicle {k}: {distance}")

        json_file_path = f"res/MIP/inst{instance}.json"
        if elapsed_time < 300:
            opt = True
        else:
            opt = False
            elapsed_time=300

        if not os.path.exists(json_file_path):
            make_json(json_file_path, ["Gurobi"], [elapsed_time], [model.objVal], [sol], [opt])

        with open(json_file_path) as json_file:
            data = json.load(json_file)
            make_json(json_file_path, ["Gurobi"], [elapsed_time], [model.objVal], [sol], [opt])