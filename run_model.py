import os
import numpy as np
import time
import asyncio
import argparse
import datetime
from datetime import timedelta
from minizinc import Instance, Model, Solver, Status

#python run_model.py --method CP --model Model_A_gc_corrected_changedoutput.mzn --instance 1,3,4 --solver gecode --timeout 20

BASE_PATH = os.getcwd()



def print_configuration(instance, model, method): 
    print("\n\n***********************************************************************")
    print("******Multiple Currier Capacitated Vehicle Routing Problem Solver******")
    print("***********************************************************************")

    print(f"Solving instance: {instance}, model: {model}, method: {method}\n")
    print("***********************************************************************\n")

def extract_data_from_dat(instance_path, verbose=True): 
    with open(instance_path, 'r') as file:
        lines = file.readlines()
    
    num_vehicles = int(lines[0].strip())
    num_clients = int(lines[1].strip())
    
    vehicles_capacity = list(map(int, lines[2].strip().split()))
    packages_size = list(map(int, lines[3].strip().split()))

    distances = [list(map(int, line.strip().split())) for line in lines[4:]]

    if verbose:
        print(f"num_vehicles: {num_vehicles}")
        print(f"num_clients: {num_clients}")
        
        if len(distances) <= 20:
            print(f"packages_size:\n {packages_size}")
            print(f"vehicles_capacity:\n {vehicles_capacity}")
            
            print(f"distances:")
            for i in distances: 
                print(i)
        else: 
            print(f"\nThe matrix has been omitted, it is too big to be printed")

    return num_vehicles, num_clients, vehicles_capacity, packages_size, distances


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


#check if the matrix is simmetric 
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


def solve_instance(model_path, solver_id, num_vehicles, num_clients, vehicles_capacity, packages_size, distances, timeout_time, int_res):
    #instanciate the model
    model = Model(model_path)
    solver = Solver.lookup(solver_id)
    instance = Instance(solver, model)

    #transform in numpy matrix
    matrix_dist=np.array(distances) 

    #compute upper and lower bound
    low_bound, min_dist_bound, up_bound = compute_bounds(distances, num_vehicles, num_clients)

    #sort the vehicle capacity list in order to implement the symmetry breaking constraints on the vehicle load
    vehicles_capacity = sorted(vehicles_capacity, reverse=True)
    
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
    timeout = timedelta(seconds=timeout_time)

    start_time = time.time()
    if int_res: 
        asyncio.run(print_intermediate_solutions(instance, timeout))
    else:
        result = instance.solve(timeout=timeout, random_seed=42)
    end_time = time.time()

    elapsed_time = end_time - start_time

    return result, elapsed_time


def get_cp_model_path(model_name):
    cp_model_path = os.path.join(BASE_PATH, "CP", "CP_Model", model_name)
    return cp_model_path

def get_mip_model_path(model_name):
    mip_model_path = os.path.join(BASE_PATH, "ILP", model_name)
    return mip_model_path

def string_to_dict(input_str):
    # Evaluate the string as a Python literal safely
    result = eval(input_str, {"datetime": datetime, "__builtins__": {}})
    return result


def timedelta_to_string(td):
    # Extract days, seconds, and microseconds
    days = td.days
    seconds = td.seconds
    microseconds = td.microseconds

    # Calculate hours, minutes, and remaining seconds
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    # Build the output string dynamically
    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if seconds > 0:
        parts.append(f"{seconds}s")
    if microseconds > 0:
        parts.append(f"{microseconds}µs")

    # Join all non-zero components with a space
    return " ".join(parts)

async def print_intermediate_solutions(instance, timeout, show_stats=False):
    last_result = None
    async for result in instance.solutions(intermediate_solutions=True, timeout=timeout):
        if result is not None and result.status is Status.SATISFIED:

            statistics_dict = string_to_dict(str(result.statistics))

            if "time" in statistics_dict.keys():
                time = timedelta_to_string(statistics_dict["time"])
                print(f"\n Intermediate Solution found at time {time}:")
                print(result.solution)

            ending_status = result.status

    if show_stats: 
        statistics_dict = string_to_dict(str(result.statistics))

        print("\n\nStatistic on search termination: \n")
        print("solveTime:  ",  statistics_dict["solveTime"])
        print("Solutions:  ", statistics_dict["solutions"])
        print("End status: ", result.status)
        print("\nBest Solution:")

    print("\n\nInstance Terminated:")
    print(ending_status)

    return last_result

#recursevely extract the route from the succ_matrix
def extract_route(row_arr, num_clients, prev=[]):
    if prev == []:
        prev = [num_clients,]
    elif row_arr[prev[-1]] == num_clients:
        prev.append(num_clients)
        return prev
    else:  
        prev.append(row_arr[prev[-1]])
    
    return extract_route(row_arr, num_clients, prev)

def reconstruct_solution(succ_matrix, num_vehicles, num_clients, distance_matrix):

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
        solution[i] = extract_route(succ_matrix[i], num_clients)


    #using the distance matrix and the solution, compute the total distance for each vehicle
    distance =[]
    for i in range(0, num_vehicles):
        total_distance = 0
        for j in range(0, len(solution[i])-1):
            total_distance += distance_matrix[solution[i][j]][solution[i][j+1]]
        distance.append(total_distance)

    return solution, distance


def solution_to_string(solution_dict, distances):
    string = ""
    for key, value in solution_dict.items():
        string += f"Vehicle {key+1} tour: (distance {distances[key]}) \n "
        string += f"{value[0]}"
        for i in value[1:]:
            string += f" -> {i}"
        string += "\n"
    return string



def main():
    parser = argparse.ArgumentParser(description="Script that takes method, model, and instance as input.")
    parser.add_argument('--method',   type=str,  required=True,  default="CP",      help='The method to use')
    parser.add_argument('--model',    type=str,  required=True,  default="Model_A", help='The model to use')
    parser.add_argument('--instance', type=str,  required=True,  default="1",       help='The instances to solve')
    parser.add_argument('--solver',   type=str,  required=False, default="gecode",  help='The solver to use')
    parser.add_argument('--timeout',  type=int,  required=False, default=300,       help='The timeout expressed in seconds')
    parser.add_argument('--int_res',  type=bool, required=False, default=False,     help='If true shows intermediate results. Buggy feature.')
    parser.add_argument('--stat',     type=bool, required=False, default=False,     help='If true shows the statistics.')

    args = parser.parse_args()
    model_name = args.model
    method = args.method
    solver_id = args.solver
    timeout_time = args.timeout
    instances = define_instances_num(args.instance)
    int_res = args.int_res
    show_stat = args.stat
    
    print_configuration(instances, args.model, args.method)

        #select the model path depending on the selected method
    if method == "CP":
        model_path = get_cp_model_path(model_name)
    elif method == "MIP":
        model_path = get_mip_model_path(model_name)
    else: 
        print("not implemented yet XD")
        exit()

    if not os.path.isfile(model_path):
        print(f"The model file can't be found. \nGiven model_ path: {model_path}\nCheck that the file exists")
        exit()

    for instance_n in instances: 

        print(f"INSTANCE: {instance_n}\n")

        instance_file_name = (f"inst{instance_n}" if instance_n >= 10 else f"inst0{instance_n}") + ".dat"
        instance_path = os.path.join(BASE_PATH, "Instances", instance_file_name)

        if not os.path.isfile(instance_path):
            print(f"The instance file can't be found. \nGiven instance_path: {instance_path}\nCheck that the file exists")
            exit()

        #extract the data from the dat file
        num_vehicles, num_clients, vehicles_capacity, packages_size, distances = extract_data_from_dat(instance_path)


        result, elapsed_time = solve_instance(model_path, solver_id, num_vehicles, num_clients, vehicles_capacity, packages_size, distances, timeout_time, int_res)

        print(f"\nFinished with state: {result.status} after {round(elapsed_time, 4)}s")

        print("\nRESULTS:")

        res_arr = str(result.solution).split("|")
        
        succ_matrix = res_arr[0]
        max_load_compute = res_arr[1]

        solution, distances = reconstruct_solution(succ_matrix, num_vehicles, num_clients, distances)


        print(solution_to_string(solution, distances))

        print(f"Max Load Compute: {max_load_compute}")
        print(f"Max Load reconstructed from sol: {max(distances)}")

        if show_stat:
            print(f"\nStatistics:\n {result.statistics}")
        
        print("\n"+"*"*50+"\n")

if __name__ == "__main__":
    main()