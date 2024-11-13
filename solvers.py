import os
import numpy as np

import time
from datetime import timedelta
from minizinc import Instance, Model, Solver, Status

import gurobipy as gb


BASE_PATH = os.getcwd()

def get_cp_model_path(model_name):
    cp_model_path = os.path.join(BASE_PATH, "CP", "CP_Model", model_name)
    return cp_model_path

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

def cp_solution_to_string(solution_dict, distances):
    string = ""
    for key, value in solution_dict.items():
        string += f"Vehicle {key+1} tour: (distance {distances[key]}) \n "
        string += f"{value[0]}"
        for i in value[1:]:
            string += f" -> {i}"
        string += "\n"
    return string

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

    #using the distance matrix and the solution, compute the total distance for each vehicle
    distance =[]
    for i in range(0, num_vehicles):
        total_distance = 0
        for j in range(0, len(solution[i])-1):
            total_distance += distance_matrix[solution[i][j]][solution[i][j+1]]
        distance.append(total_distance)

    return solution, distance

def check_weights(packages_size, vehicles_capacity, solution):

    loads = [sum([packages_size[i] for i in sol]) for sol in solution]


    vc = vehicles_capacity.copy() 
    sol = solution.copy() 
    for _ in range(len(vehicles_capacity)): 
        #take the index of the max element in the vehicles_capacity list
        max_index = vc.index(max(vc))
        #take the index of the max element in the loads list
        max_load = loads.index(max(loads))
        sol[max_index] = solution[max_load]
        
        print("loads", loads)
        print("max_load", max_load)
        print("vc", vc)
        print("max_index", max_index)

        #put the max load to 0 in order to avoid to take it again
        loads[max_load] = 0
        vc[max_index] = 0

    return sol

def get_gurobi_env():
    """LICENSE"""
    # LICENSE FOR ACADEMIC VERSION OF GUROBI
    # Create an environment with your WLS license
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

def reconstruct_gurobi_solution(x):
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


    print(cp_solution_to_string(solution, distances))

    print(f"Max distance Compute: {max_dist_compute}")
    print(f"Max distance reconstructed from sol: {max(distances)}")
    
    print("\n"+"*"*50+"\n")
    solution = list([sol for sol in solution.values()])
    
    #for each element in the solution, convert it to a list and each element to an int
    solution = [list(map(int, sol)) for sol in solution]

    #delete the firt and last element for aeach list in the solution, in order to have only the clients
    solution = [sol[1:-1] for sol in solution]

    #TO DO, if we use the symmetry breaking constraint, we need to check the weights of the vehicles
    #solution = check_weights(packages_size, vehicles_capacity, solution)

    #add one to each element in the solution to have the correct index
    solution = [[sol+1 for sol in s] for s in solution]
    
    print(solution)

    return {"time": solver_time+preprocessing_time, "optimal": result.status == Status.OPTIMAL_SOLUTION, "obj": max_dist_compute, "sol": solution}

def solve_ilp_guroby(instance_data, timeout_time):

    distances, num_vehicles, num_clients, vehicles_capacity, packages_size = instance_data["distances"], instance_data["num_vehicles"], instance_data["num_clients"], instance_data["vehicles_capacity"], instance_data["packages_size"]

    #transform in numpy matrix
    matrix_dist=np.array(distances) 
    #compute upper and lower bound
    start_time = time.time()
    #low_bound, min_dist_bound, up_bound = compute_bounds(distances, num_vehicles, num_clients)
    low_bound, up_bound, min_dist_bound = calculate_mccrp_bounds(matrix_dist) 
    end_time = time.time()
    preprocessing_time = end_time - start_time

    CUSTOMERS = list(range(1,num_clients+1))
    NODES = list(range(0, num_clients+1)) 
    COURIERS = list(range(1,num_vehicles+1))

    # Create an environment with your WLS license

    env = get_gurobi_env()
    model = gb.Model(name="MCCVRP",env=env)
    x = model.addVars(NODES,NODES,COURIERS, vtype=gb.GRB.BINARY, name="x_ijk")
    y = model.addVars(NODES,COURIERS, vtype=gb.GRB.BINARY, name="y_ik")
    d = model.addVars(CUSTOMERS, COURIERS, vtype=gb.GRB.CONTINUOUS, name="d_ik")
    max_distance = model.addVar(name='max_distance')
    model.setObjective(max_distance, sense=gb.GRB.MINIMIZE)
    
    for k in COURIERS:
        model.addConstr(gb.quicksum(x[i,j,k]*distances[i-1][j-1] for i in NODES for j in NODES) <= max_distance)

    # CONSTRAINTS 
    #Upper e lower bound constraints
    model.addConstr(max_distance <= up_bound)
    model.addConstr(max_distance >= low_bound)

    #all the curriers are used
    model.addConstr(gb.quicksum(y[0,k] for k in COURIERS)==num_vehicles)
    
    for i in CUSTOMERS:
        #each item is assigned to only one courier
        model.addConstr(gb.quicksum(y[i,k] for k in COURIERS)==1)
        #starting from a node every customer must go to an another node
        model.addConstr(gb.quicksum(x[i,j,k] for j in NODES for k in COURIERS)==1)

    for j in CUSTOMERS:
        #each customer has exactly one predecessor
        model.addConstr(gb.quicksum(x[i,j,k] for i in NODES for k in COURIERS)==1)

    for k in COURIERS:
        #the sum of the packages assigned to each courier must be less than the capacity of the vehicle
        model.addConstr(gb.quicksum(y[i,k]*packages_size[i-1] for i in CUSTOMERS)<=vehicles_capacity[k-1])
        #the main diagonal of x must be 0, since a customer can't be the predecessor of itself 
        model.addConstr(gb.quicksum(x[j,j,k] for j in NODES)==0)
        #each currier must go back to the depot
        model.addConstr(gb.quicksum(x[i,0,k] for i in NODES)==1)

    for i in CUSTOMERS:
        for k in COURIERS:
            #if a currier is covering a route i to j, then the currier is also covering the route j to i
            model.addConstr(gb.quicksum(x[i,j,k] for j in NODES)==gb.quicksum(x[j,i,k] for j in NODES))
            #if a currier is covering a route i to j, then we assign the item to the vehicle in the decision variable y 
            model.addConstr(gb.quicksum(x[j,i,k] for j in NODES)==y[i,k])

#Bho sembra stupido lol
#    for k in COURIERS:
#        for j in NODES:
#            model.addConstr(gb.quicksum(x[i,j,k] for i in NODES) == gb.quicksum(x[i,j,k] for i in NODES))

    #Subtour elimination using MTZ formulation
    for k in COURIERS:
        for i in CUSTOMERS:
            for j in CUSTOMERS:
                if i != j:
                    model.addConstr(d[i, k] - d[j, k] + num_clients * x[i, j, k] <= num_clients - 1)

#sembra ridondante, non si capisce dove prende la k
#    for i in CUSTOMERS:
#        for j in CUSTOMERS:
#            if i != j:
#                model.addConstr(d[i, k] - d[j , k] + num_clients * x[i, j, k] <= num_clients - 1)

    # Set the time limit 
    time_limit = timeout_time - preprocessing_time
    model.setParam(gb.GRB.Param.TimeLimit, time_limit)

    start_time = time.time()
    model.optimize()
    end_time = time.time()
    solver_time = end_time - start_time
    
    print("SOLUTION FOUND ", model.objVal)

        # Check the optimization status and retrieve the solution if available
    print("#"*50)
    print(f"\nFinished with state: {model.status} after {round(solver_time, 4)}s, preprocessing time: {round(preprocessing_time, 4)}s")
    print("RESULTS:")
    print("Objective value: ", model.objVal)

    routes = reconstruct_gurobi_solution(x)
    
    for k, route in sorted(routes.items(), key=lambda item: item[0]):
            print(f"Courier {k} route: {route}")
    
    return {"time": model.Runtime, "optimal": model.status == gb.GRB.OPTIMAL, "obj": model.objVal, "sol": []}




