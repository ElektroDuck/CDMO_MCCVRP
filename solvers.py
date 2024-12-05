import os
import math
import json
import numpy as np
import subprocess

import time
from datetime import timedelta
from minizinc import Instance, Model, Solver, Status

import gurobipy as gb

import SAT.sat_model as sat_model

BASE_PATH = os.getcwd()

def get_cp_model_path(model_name):
    cp_model_path = os.path.join(BASE_PATH, "CP", "CP_Model", model_name)
    return cp_model_path

def get_ilp_model_path(model_name):
    ilp_model_path = os.path.join(BASE_PATH, "ILP", model_name)
    return ilp_model_path

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

def calculate_mccrp_bounds(distance_matrix):
    """
    Calculate approximate lower and upper bounds for MCCRP using distance matrix.
    
    Parameters:
    distance_matrix (numpy.ndarray): A matrix where element [i][j] is the distance between node i and node j.
    
    Returns:
    tuple: (lower_bound, upper_bound, min_dist_bound)
    """
    
    # 1. Lower Bound Calculation 
    matrix_dist = np.array(distance_matrix) #transform in numpy matrix
    last_row = matrix_dist[-1, :]  # selects the last row
    last_column = matrix_dist[:, -1]  # selects the last column
    result = last_row + last_column
    lower_bound = max(result)
    
    # 2. compute the minimum distance each vehicle has to travel
    min_dist_bound = min(result) 

    # 3. Upper Bound Calculation using a Greedy Heuristic
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
    
    return lower_bound, upper_bound, min_dist_bound

def check_simmetry(d):
    """
    Check if the matrix is simmetric.
    This is done comparing the distace matrix with its transpose
    The comparison returns an Matrix of bool, where c[i,j] = True if d[i,j] == dt[i,j]
    If the sum of the content of the matrix c = elements contained in d, then it means that the matrix is symmetric
    """
    n, m = np.shape(d)
    dt = np.transpose(d)
    return np.sum(dt==d) == (n*m)

def solution_to_string(solution_dict, distances):
    
    string = ""
    for key, value in solution_dict.items():
        string += f"Vehicle {key+1} tour: (distance {distances[key]}) \n "
        string += f"{value[0]}"
        for i in value[1:]:
            string += f" -> {i}"
        string += "\n"
    return string

def compute_distances(distance_matrix, solution, num_vehicles):
    #using the distance matrix and the solution, compute the total distance for each vehicle
    distance =[]
    keys = list(solution.keys())
    start, end = min(keys), max(keys)+1

    for i in range(start, end):
        total_distance = 0
        for j in range(0, len(solution[i])-1):
            total_distance += distance_matrix[solution[i][j]][solution[i][j+1]]
        distance.append(total_distance)

    return distance

def cp_extract_route(row_arr, num_clients, prev=[]):
    if prev == []:
        prev = [num_clients,]
    elif row_arr[prev[-1]] == num_clients:
        prev.append(num_clients)
        return prev
    else:  
        prev.append(row_arr[prev[-1]])
    
    return cp_extract_route(row_arr, num_clients, prev)

def reconstruct_cp_solution(succ_matrix, num_vehicles, num_clients, distance_matrix):

    #take the string succ_matrix, transform it as an array, then reshape it as a matrix
    succ_matrix = succ_matrix.replace("[", "").replace("]", "")
    succ_matrix = np.array(succ_matrix.split(",")).reshape(num_vehicles, num_clients+1)

    #convert the content of the matrix to the int type
    succ_matrix = succ_matrix.astype(int)

    #succ_matrix is composed by number that span from 1 to n_vehicles+1
    #to have the correct index, we subtract 1 to each element
    succ_matrix =  succ_matrix - 1
    
    solution = {}

    #for each row, take the index of the elemnent that dosn't correspond with the index of the column
    for i in range(0, num_vehicles):
        #extract the route from the succ_matrix and store it in the solution dictionary
        solution[i] = cp_extract_route(succ_matrix[i], num_clients)

    distance = compute_distances(distance_matrix, solution, num_vehicles)

    return solution, distance

def ilp_extract_route(succ_matrix, prev_idx, prev=[]):

    #retrive index of the max element in the array at the position next_idx
    next_idx = int(np.argmax(succ_matrix[prev_idx]))
    if prev == []:
        prev = [len(succ_matrix)-1]
    elif prev_idx == len(succ_matrix)-1:
        return prev
    
    prev.append(next_idx)

    return ilp_extract_route(succ_matrix, next_idx, prev)

def reconstruct_ilp_minizinc_solution(succ_matrix, num_vehicles, num_clients, distance_matrix):

    succ_matrix = succ_matrix.replace("[", "").replace("]", "").replace(" ", "").replace("\"", "")
    succ_matrix = np.array(succ_matrix.split(",")).reshape(num_clients+1, num_clients+1, num_vehicles)
    succ_matrix = succ_matrix.astype(int)

    solution = {}
    for i in range(0, num_vehicles):
        solution[i] = ilp_extract_route(succ_matrix[:,:,i], len(succ_matrix)-1)

    return solution

def get_gurobi_env():
    # This is my (Luca Tedeschini) personal Gurobi license
    # Use it just to test our model only
    params = {
    "WLSACCESSID": '216dd889-fa22-449b-a888-e35218548ac7',
    "WLSSECRET": 'c4660e0b-df0a-49a3-b518-0791ddc21565',
    "LICENSEID": 2581103,
    }
    env = gb.Env(params=params)
    return env

def retrieve_elements(middle,first):
    for i in middle:
        if i[0]==first:
            return i

def reconstruct_gurobi_solution(x, num_vehicles, distance_matrix, n_clients):
    routes = {}
    for (i, j, k) in x:
        if x[(i, j, k)].x == 1:
            if k not in routes:
                routes[k] = [(i, j)]
            else:
                routes[k].append((i, j))
    
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

    for k in routes:
        routes[k] = [t[0] for t in routes[k]]
        routes[k] = [i-1 for i in routes[k]]
        routes[k][0] = n_clients
        routes[k] = routes[k] + [n_clients]
    
    solution = {}
    for k in routes:
        solution[k-1] = routes[k]

    distances = compute_distances(distance_matrix, solution, num_vehicles)

    return solution, distances

def solve_cp(model_name, solver_id, instance_data, timeout_time): 
    model_path = get_cp_model_path(model_name)

    #instanciate the model
    model = Model(model_path)
    solver = Solver.lookup(solver_id)
    instance = Instance(solver, model)
    
    distances, num_vehicles, num_clients, vehicles_capacity, packages_size = instance_data["distances"], instance_data["num_vehicles"], instance_data["num_clients"], instance_data["vehicles_capacity"], instance_data["packages_size"]

    #transform in numpy matrix
    matrix_dist=np.array(distances) 
    #compute upper and lower bound
    start_time = time.time()
    #low_bound, min_dist_bound, up_bound = compute_bounds(distances, num_vehicles, num_clients)
    low_bound, up_bound, min_dist_bound = calculate_mccrp_bounds(matrix_dist) 
    end_time = time.time()
    preprocessing_time = end_time - start_time

    #sort the vehicle capacity list in order to implement the symmetry breaking constraints on the vehicle load
    #vehicles_capacity = sorted(vehicles_capacity, reverse=True)
    
    #assign the input variable to the minizinc variable
    instance["num_vehicles"] = num_vehicles
    instance["num_clients"] = num_clients
    instance["size"] = packages_size
    instance["capacity"] = vehicles_capacity
    instance["distances"] = distances
    instance["low_bound"] = low_bound
    instance["up_bound"] = up_bound
    instance["min_dist_bound"] = min_dist_bound

    #if there is a symmetry in the matrix, add a symmetry breaking constraint
    if check_simmetry(matrix_dist):
        model.add_string("constraint forall(j in vehicles) (successor[j,num_clients+1]<arg_max(successor[j,..]));")
        print("\nThe matrix is symmetric, a symmetry breaking constrain has been added\n")

    #solve the problem
    timeout = timedelta(seconds=(timeout_time-preprocessing_time))

    start_time = time.time()
    result = instance.solve(timeout=timeout, random_seed=42)
    end_time = time.time()

    solver_time = end_time - start_time

    print(f"\nFinished with state: {result.status} after {round(solver_time, 4)}s, preprocessing time: {round(preprocessing_time, 4)}s\n")
    print("\nRESULTS:")

    res_arr = str(result.solution).split("|")
    
    if result.status is Status.UNKNOWN or result.status is Status.UNSATISFIABLE or result.status is Status.ERROR:
        print("No solution found, exit status: ", result.status)  
        return {"time": 300, "optimal": False, "obj": 0, "sol": []}


    succ_matrix = res_arr[0]
    max_dist_compute = res_arr[1]

    solution, distances = reconstruct_cp_solution(succ_matrix, num_vehicles, num_clients, distances)


    print(solution_to_string(solution, distances))

    print(f"Max distance Compute: {max_dist_compute}")
    print(f"Max distance reconstructed from sol: {max(distances)}")
    
    print("\n"+"*"*50+"\n")
    #solution = list([sol for sol in solution.values()])
    solution = list([solution[i] for i in sorted(solution.keys())])

    #for each element in the solution, convert it to a list and each element to an int
    solution = [list(map(int, sol)) for sol in solution]

    #delete the firt and last element for aeach list in the solution, in order to have only the clients
    solution = [sol[1:-1] for sol in solution]

    #add one to each element in the solution to have the correct index
    solution = [[sol+1 for sol in s] for s in solution]

    total_time = solver_time+preprocessing_time

    #Since the solver dosn't stop exactly at the given second ensure the constraint sol_not_optimal -> time = timeout_time
    if result.status is not Status.OPTIMAL_SOLUTION:
        total_time = timeout_time

    return {"time": total_time, "optimal": result.status == Status.OPTIMAL_SOLUTION, "obj": max_dist_compute, "sol": solution}

def solve_ilp_minizinc(model_name, solver_id, instance_data, timeout_time):
    model_path = get_ilp_model_path(model_name)
    
    model = Model(model_path)
    solver = Solver.lookup(solver_id)
    instance = Instance(solver, model)

    distances, num_vehicles, num_clients, vehicles_capacity, packages_size = instance_data["distances"], instance_data["num_vehicles"], instance_data["num_clients"], instance_data["vehicles_capacity"], instance_data["packages_size"]
    #transform in numpy matrix
    matrix_dist=np.array(distances) 
    #compute upper and lower bound
    start_time = time.time()
    #low_bound, min_dist_bound, up_bound = compute_bounds(distances, num_vehicles, num_clients)
    low_bound, up_bound, min_dist_bound = calculate_mccrp_bounds(matrix_dist) 
    end_time = time.time()
    preprocessing_time = end_time - start_time

    instance["num_vehicles"] = num_vehicles
    instance["num_clients"] = num_clients
    instance["size"] = packages_size
    instance["capacity"] = vehicles_capacity
    instance["distances"] = distances

    instance["low_bound"] = low_bound
    instance["up_bound"] = up_bound
    instance["min_dist_bound"] = min_dist_bound


    timeout = timedelta(seconds=(timeout_time-preprocessing_time))
    start_time = time.time()
    result = instance.solve(timeout=timeout, random_seed=42)
    end_time = time.time()

    solver_time = end_time - start_time

    print(f"\nFinished with state: {result.status} after {round(solver_time, 4)}s, preprocessing time: {round(preprocessing_time, 4)}s\n")
    print("\nRESULTS:")
    
    if result.status is Status.UNKNOWN or result.status is Status.UNSATISFIABLE or result.status is Status.ERROR:
        print("No solution found, exit status: ", result.status)  
        return {"time": 300, "optimal": False, "obj": 0, "sol": []}


    res = str(result.solution)

    max_dist_compute = int(res.split("|")[0].replace("\"", ""))
    succ_matrix = res.split("|")[1]

    print(f"Max distance Compute: {max_dist_compute}")

    solution = reconstruct_ilp_minizinc_solution(succ_matrix, num_vehicles, num_clients, matrix_dist)
    vehicle_distances = compute_distances(distances, solution, num_vehicles)
    print("Max distance (rec from sol): ", max(vehicle_distances))

    print("routes: ")
    print(solution_to_string(solution, vehicle_distances))

    #solution = list([sol for sol in solution.values()])
    solution = list([solution[i] for i in sorted(solution.keys())])

    #for each element in the solution, convert it to a list and each element to an int
    solution = [list(map(int, sol)) for sol in solution]

    #delete the firt and last element for aeach list in the solution, in order to have only the clients
    solution = [sol[1:-1] for sol in solution]

    #add one to each element in the solution to have the correct index
    solution = [[sol+1 for sol in s] for s in solution]

    total_time = solver_time+preprocessing_time

    #Since the solver dosn't stop exactly at the given second ensure the constraint sol_not_optimal -> time = timeout_time
    if result.status is not Status.OPTIMAL_SOLUTION:
        total_time = timeout_time

    return {"time": total_time, "optimal": result.status == Status.OPTIMAL_SOLUTION, "obj": max_dist_compute, "sol": solution}

def solve_mip_gurobi(instance_data, timeout_time):
    distances, num_vehicles, num_clients, vehicles_capacity, packages_size = instance_data["distances"], instance_data["num_vehicles"], instance_data["num_clients"], instance_data["vehicles_capacity"], instance_data["packages_size"]

    #transform in numpy matrix
    matrix_dist=np.array(distances)

    #Timer for preprocessing
    start_time = time.time()

    #low_bound, min_dist_bound, up_bound = compute_bounds(distances, num_vehicles, num_clients)
    low_bound, up_bound, _ = calculate_mccrp_bounds(matrix_dist)

    end_time = time.time()

    preprocessing_time = end_time - start_time

    # Parameter definition
    CLIENT = list(range(1,num_clients+1))
    NODES = list(range(0, num_clients+1)) 
    VEHICLES = list(range(1,num_vehicles+1))

    # Create an environment with your WLS license
    env = get_gurobi_env()
    model = gb.Model(name="MCCVRP",env=env)

    # Create model variables
    x = model.addVars(NODES,NODES,VEHICLES, vtype=gb.GRB.BINARY, name="x_ijk")
    y = model.addVars(NODES,VEHICLES, vtype=gb.GRB.BINARY, name="y_ik")
    d = model.addVars(CLIENT, VEHICLES, vtype=gb.GRB.CONTINUOUS, name="d_ik")
    max_distance = model.addVar(name='max_distance')
    model.setObjective(max_distance, sense=gb.GRB.MINIMIZE)



    # CONSTRAINTS
    # Constraint I & II: bounds on objective variable
    model.addConstr(max_distance <= up_bound)
    model.addConstr(max_distance >= low_bound)
    
    for i in CLIENT:
        # Constraint III: Each client has an unique vehicle assigned to it
        model.addConstr(gb.quicksum(y[i,k] for k in VEHICLES)==1)
        # Constraint IV: Each client must be visited exactly once across all vehicles.
        model.addConstr(gb.quicksum(x[i,j,k] for j in NODES for k in VEHICLES)==1)

    for j in CLIENT:
        # Constrain V: There must be a way to reach the node you are currently standing on (Reduntant but reduce search space)
        model.addConstr(gb.quicksum(x[i,j,k] for i in NODES for k in VEHICLES)==1)

    for k in VEHICLES:
        # Constraint VI: Capacity constraint (Reduntant but reduce search space)
        model.addConstr(gb.quicksum(y[i,k]*packages_size[i-1] for i in CLIENT)<=vehicles_capacity[k-1])
        # Constraint VII: Diagonal = 0 (avoid loops)
        model.addConstr(gb.quicksum(x[j,j,k] for j in NODES)==0)
        # Constraint VIII: Last point is the depot
        model.addConstr(gb.quicksum(x[i,0,k] for i in NODES)==1)
    

    for i in CLIENT:
        for k in VEHICLES:
            # Constraint IX: If a courier enters a node, then it must exit that node
            model.addConstr(gb.quicksum(x[i,j,k] for j in NODES)==gb.quicksum(x[j,i,k] for j in NODES))
            # Constraint X: If a courier is covering a client, we assign the item to it
            model.addConstr(gb.quicksum(x[j,i,k] for j in NODES)==y[i,k])
    

    # Constraint XI: Subtour elimination using MTZ formulation
    for k in VEHICLES:
        for i in CLIENT:
            for j in CLIENT:
                if i != j:
                    model.addConstr(d[i, k] - d[j, k] + num_clients * x[i, j, k] <= num_clients - 1)

    for k in VEHICLES:
        # Objective function
        model.addConstr(gb.quicksum(x[i,j,k]*distances[i-1][j-1] for i in NODES for j in NODES) <= max_distance)




    # Set the time limit 
    time_limit = timeout_time - preprocessing_time
    model.setParam(gb.GRB.Param.TimeLimit, time_limit)

    # Set the model parameters
    #model.setParam('Method', 2)
    model.setParam('MIPFocus', 1)
    model.setParam('ImproveStartTime', 200)
    model.setParam('Presolve', 1)
    model.setParam('Cuts', 1)

    start_time = time.time()
    model.optimize()
    end_time = time.time()
    solver_time = end_time - start_time
    
    print("SOLUTION FOUND ", model.objVal)

    # Check the optimization status and retrieve the solution if available
    print("#"*50)
    print(f"\nFinished with state: {model.status} after {round(solver_time, 4)}s, preprocessing time: {round(preprocessing_time, 4)}s")
    print("RESULTS:")
    print("max distance: ", model.objVal)

    # Check if no solution was found
    if model.objVal == math.inf:
        return {"time": timeout_time, "optimal": False, "obj": 0, "sol": []}


    routes, distances = reconstruct_gurobi_solution(x, num_vehicles, distances, num_clients)

    print(solution_to_string(routes, distances))

    solution = list([routes[i] for i in sorted(routes.keys())])
    # Retrieve as a List the solution
    solution = [list(map(int, sol)) for sol in solution]
    # Remove the depots from the solutions
    solution = [sol[1:-1] for sol in solution]
    # Re-index the solution to be compliant with the solution checker
    solution = [[sol+1 for sol in s] for s in solution]

    sol_time = model.Runtime + preprocessing_time
    

    # Set the solution time to 300 if no optimal solution are found
    if model.status != gb.GRB.OPTIMAL:
        sol_time = timeout_time

    return {"time": sol_time, "optimal": model.status == gb.GRB.OPTIMAL, "obj": model.objVal, "sol": solution}

def solve_mip(instance_data, timeout_time, model, solver="gecode"):
    if model == "gurobi":
        return solve_mip_gurobi(instance_data, timeout_time)
    elif model == "minizinc":
        return solve_ilp_minizinc("ILP_model.mzn", solver, instance_data, timeout_time)
    else:
        raise ValueError("Specified MIP Model not recognized! \n Please choose between 'gurobi' and 'minizinc'")

def solve_smt(instance_data, instance_n, timeout_time):
    flag = False
    try:
        # Calling the SMT solver as a subprocess, so we can apply timeout
        subprocess.call(["python3", "./SMT/smt_subprocess.py", str(instance_n)], timeout=timeout_time)
    except subprocess.TimeoutExpired:
        print("Solution TIMEOUT")
        flag = True

    # Saving results
    try:
        with open("tmp_output.json", "r") as f:
            file = f.readlines()[0]
        result_dict = json.loads(file)
        if flag: 
            result_dict["time"] = 300
            result_dict["optimal"] = False
        
        sol = result_dict["sol"]
        print_sol = [[instance_data['num_clients']] + sublist + [instance_data['num_clients']]  for sublist in sol]
        
        print_sol = {key: value for key, value in enumerate(print_sol)}

        distances = compute_distances(instance_data["distances"], print_sol, instance_data["num_vehicles"])
        print(solution_to_string(print_sol, distances))

        for i in range(len(sol)):
            sol[i] = [x+1 for x in sol[i]]
    except:
        result_dict = {
            "time" : 300,
            "optimal" : False,
            "obj" : 0,
            "sol" : []
        }

    #print results 
    print("\n"+"*"*50+"\n")
    print("Optimal solution: ", result_dict["optimal"])
    print("Objective function value: ", result_dict["obj"])
    print("Time: ", result_dict["time"])

    if os.path.exists("tmp_output.json"):
        os.remove("tmp_output.json")

    return result_dict

def solve_sat(instance_data, instance_n, timeout_time):
    distances, num_vehicles, num_clients, vehicles_capacity, packages_size = instance_data["distances"], instance_data["num_vehicles"], instance_data["num_clients"], instance_data["vehicles_capacity"], instance_data["packages_size"]
    low_bound, up_bound, _ = calculate_mccrp_bounds(distances)
    sat_result =  sat_model.sat_model(num_vehicles, num_clients, vehicles_capacity, packages_size, distances, up_bound, low_bound, display_solution=True, timeout_duration=timeout_time, search="Binary")
    result = {
        "time": sat_result[1],
        "optimal": (not sat_result[1]>=timeout_time), #if the time is less than the timeout, then the solution is considered optimal
        "obj": sat_result[0],
        "sol": sat_result[2]
    }
    return result