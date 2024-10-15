import os
import numpy
import asyncio
import argparse
<<<<<<< HEAD
from minizinc import Instance, Model, Solver

#python run_model.py --method method1 --model model_A.mzn --solver gecode --instance 1 
=======
import datetime
from datetime import timedelta
from minizinc import Instance, Model, Solver, Status

#python --method CP --model Model_A.mzn --instance 1,3,4 --solver chuffed --timeout 20
>>>>>>> b5eb065f64bcf4c841f2a88f8ba55fbdb9350742

BASE_PATH = os.getcwd()

def define_instances_num(instances): 

    if instances == "all":
        instances =  [i for i in range(1,22)] 
    else:
        instances = instances.split(",")
        instances = [int(i) for i in instances]
    return instances

def print_configuration(instance, model, method): 
    print(f"solving instance: {instance}, model: {model}, method: {method}")

<<<<<<< HEAD
def extract_data_from_dat(instance_path, verbouse=False): 
=======
def extract_data_from_dat(instance_path, verbose=True): 
>>>>>>> b5eb065f64bcf4c841f2a88f8ba55fbdb9350742
    with open(instance_path, 'r') as file:
        lines = file.readlines()
    
    num_vehicles = int(lines[0].strip())
    num_clients = int(lines[1].strip())
    
    vehicles_capacity = list(map(int, lines[2].strip().split()))
    packages_size = list(map(int, lines[3].strip().split()))

    distances = [list(map(int, line.strip().split())) for line in lines[4:]]

<<<<<<< HEAD

    if verbouse: 
=======
    if verbose:
>>>>>>> b5eb065f64bcf4c841f2a88f8ba55fbdb9350742
        print(f"num_vehicles: {num_vehicles}")
        print(f"num_clients: {num_clients}")
        print(f"packages_size:\n {packages_size}")
        print(f"vehicles_capacity:\n {vehicles_capacity}")
<<<<<<< HEAD
    
        print(f"distances:")
        for i in distances: 
            print(i)
    
=======
        
        print(f"distances:")
        for i in distances: 
            print(i)

>>>>>>> b5eb065f64bcf4c841f2a88f8ba55fbdb9350742
    return num_vehicles, num_clients, vehicles_capacity, packages_size, distances


def solve_instance(model_path, solver_id, num_vehicles, num_clients, vehicles_capacity, packages_size, distances, timeout_time):
    #instanciate the model
    model = Model(model_path)
    solver = Solver.lookup(solver_id)
    instance = Instance(solver, model)

    #assign the input variable to the minizinc variable
    instance["num_vehicles"] = num_vehicles
    instance["num_clients"] = num_clients
    instance["size"] = packages_size
    instance["capacity"] = vehicles_capacity
    instance["distances"] = distances

    #crea una funzione apposita per upper e lowe boud
    matrix_dist=numpy.array(distances) #transform in numpy matrix
    last_row = matrix_dist[-1, :]  # selects the last row
    last_column = matrix_dist[:, -1]  # selects the last column
    result = last_row + last_column
    low_bound = max(result)
    up_bound = sum(result)
    instance["low_bound"] = low_bound
    instance["up_bound"] = up_bound

    #solve the problem
    timeout = timedelta(seconds=timeout_time)
    result = asyncio.run(print_intermediate_solutions(instance, timeout))
    return result

def get_cp_model_path(model_name):

    cp_model_path = os.path.join(BASE_PATH, "CP", "CP_Model", model_name)
    #check if the files exists
    if not os.path.isfile(cp_model_path):
        print(f"The model file can't be found. \nGiven model_ path: {cp_model_path}\nCheck that the file exists")
        exit()

    return cp_model_path

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


async def print_intermediate_solutions(instance, timeout):
    last_result = None
    async for result in instance.solutions(intermediate_solutions=True, timeout=timeout):
        if result is not None and result.status is Status.SATISFIED:

            statistics_dict = string_to_dict(str(result.statistics))

            if "time" in statistics_dict.keys():
                time = timedelta_to_string(statistics_dict["time"])
                print(f"\n Intermediate Solution found at time {time}:")
                print(result.solution)
                print("\nStatistics")
                print(result.statistics)
                print("\n")


    statistics_dict = string_to_dict(str(result.statistics))

    print("\n\nStatistic on search termination: \n")
    print("solveTime:  ",  statistics_dict["solveTime"])
    print("Solutions:  ", statistics_dict["solutions"])
    print("End status: ", result.status)

    print("\nBest Solution:")

    ending_status = result.status

    print("\n\nInstance Terminated:")
    print(ending_status)

    return last_result


def main():
    parser = argparse.ArgumentParser(description="Script that takes method, model, and instance as input.")
<<<<<<< HEAD
    parser.add_argument('--method', type=str, required=True, help='The method to use')
    parser.add_argument('--model', type=str, required=True, help='The model to use')
    parser.add_argument('--instance', type=int, required=True, help='The instance to use')
    parser.add_argument('--solver', type=str, required=True, help='The solver to use', default='gecode')
=======
    parser.add_argument('--method',     type=str, required=True, default="CP",      help='The method to use')
    parser.add_argument('--model',      type=str, required=True, default="Model_A", help='The model to use')
    parser.add_argument('--instance',   type=str, required=True, default="1",         help='The instances to solve')
    parser.add_argument('--solver',     type=str, required=True, default="gecode",  help='The solver to use')
    parser.add_argument('--timeout',    type=int, required=True, default=300,       help='The timeout expressed in seconds')

>>>>>>> b5eb065f64bcf4c841f2a88f8ba55fbdb9350742

    args = parser.parse_args()
    model_name = args.model
    method = args.method
<<<<<<< HEAD
    solver = args.solver

    print_configuration(args.instance, args.model, args.method)

    model_path = CP_MODELS_FOLDER + model_name
    instance_path = INSTANCE_FOLDER + (f"inst{instance_num}" if instance_num >= 10 else f"inst0{instance_num}") + ".dat"
=======
    solver_id = args.solver
    timeout_time = args.timeout
    instances = define_instances_num(args.instance)
>>>>>>> b5eb065f64bcf4c841f2a88f8ba55fbdb9350742
    
    print_configuration(instances, args.model, args.method)

        #select the model path depending on the selected method
    if method == "CP":
        model_path = get_cp_model_path(model_name)
    else: 
        print("not implemented yet XD")
        exit()

    for instance_n in instances: 
        instance_file_name = (f"inst{instance_n}" if instance_n >= 10 else f"inst0{instance_n}") + ".dat"
        instance_path = os.path.join(BASE_PATH, "Instances", instance_file_name)

<<<<<<< HEAD
    model = Model(model_path)
    solver = Solver.lookup(solver)
    instance = Instance(solver, model)
    instance["num_vehicles"] = num_vehicles
    instance["num_clients"] = num_clients
    instance["capacity"] = vehicles_capacity
    instance["size"] = packages_size
    instance["distances"] = distances

    result = instance.solve()
    print(result)
=======
        if not os.path.isfile(instance_path):
            print(f"The instance file can't be found. \nGiven instance_path: {instance_path}\nCheck that the file exists")
            exit()

        #extract the data from the dat file
        num_vehicles, num_clients, vehicles_capacity, packages_size, distances = extract_data_from_dat(instance_path)

        try:
            result = solve_instance(model_path, solver_id, num_vehicles, num_clients, vehicles_capacity, packages_size, distances, timeout_time)
        except: 
            pass

        print("FINITO, RISULTATO: ")
        print(result)
        print("*"*50)
>>>>>>> b5eb065f64bcf4c841f2a88f8ba55fbdb9350742

if __name__ == "__main__":
    main()