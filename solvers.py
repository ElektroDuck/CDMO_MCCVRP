import os
import numpy as np

import time
from datetime import timedelta

from minizinc import Instance, Model, Solver, Status


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


def solve_cp(model_name, solver_id, instance_data, timeout_time, int_res): 
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
    low_bound, min_dist_bound, up_bound = compute_bounds(distances, num_vehicles, num_clients)
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
