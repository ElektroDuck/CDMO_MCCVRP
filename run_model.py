import os
import json
import asyncio
import argparse
from datetime import timedelta
from minizinc import Instance, Model, Solver

#python run_model.py --method cp --model Model_A.mzn --instance 1 --solver chuffed

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
        print(f"packages_size:\n {packages_size}")
        print(f"vehicles_capacity:\n {vehicles_capacity}")
        
        print(f"distances:")
        for i in distances: 
            print(i)

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

    #solve the problem
    timeout = timedelta(seconds=timeout_time)
    asyncio.run(print_intermediate_solutions(instance, timeout))

def get_cp_model_path(model_name):

    cp_model_path = os.path.join(BASE_PATH, "CP", "CP_Model", model_name)
    #check if the files exists
    if not os.path.isfile(cp_model_path):
        print(f"The model file can't be found. \nGiven model_ path: {cp_model_path}\nCheck that the file exists")
        exit()

    return cp_model_path


async def print_intermediate_solutions(instance, timeout=300):
    async for result in instance.solutions(intermediate_solutions=True, timeout=timeout):
        print(f"\nIntermediate Solution:")
        print(result.solution)
        
    return result.statistics


def main():
    parser = argparse.ArgumentParser(description="Script that takes method, model, and instance as input.")
    parser.add_argument('--method',     type=str, required=True, default="CP",      help='The method to use')
    parser.add_argument('--model',      type=str, required=True, default="Model_A", help='The model to use')
    parser.add_argument('--instance',   type=str, required=True, default="1",         help='The instances to solve')
    parser.add_argument('--solver',     type=str, required=True, default="gecode",  help='The solver to use')
    parser.add_argument('--timeout',    type=int, required=True, default=300,       help='The timeout expressed in seconds')


    args = parser.parse_args()
    model_name = args.model
    method = args.method
    solver_id = args.solver
    timeout_time = args.timeout
    instances = define_instances_num(args.instance)
    
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

        if not os.path.isfile(instance_path):
            print(f"The instance file can't be found. \nGiven instance_path: {instance_path}\nCheck that the file exists")
            exit()

        #extract the data from the dat file
        num_vehicles, num_clients, vehicles_capacity, packages_size, distances = extract_data_from_dat(instance_path)


        sol_stat = solve_instance(model_path, solver_id, num_vehicles, num_clients, vehicles_capacity, packages_size, distances, timeout_time)

if __name__ == "__main__":
    main()