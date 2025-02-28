import os
import math
import numpy as np
import argparse
import solvers
import json

#Test: 
#python main.py --method CP --model Model_A_gc_corrected_changedoutput.mzn --instance 1-10 --solver gecode --timeout 20 --update_json True

BASE_PATH = os.getcwd()


def define_instances_num(instances): 
    if instances == "all":
        instances =  [i for i in range(1,22)] 
    elif "-" in instances:
        limits = instances.split("-")
        #print(limits, int(limits[0]), int(limits[1])+1)
        instances = [i for i in range(int(limits[0]), int(limits[1])+1)]
    else:
        instances = instances.split(",")
        instances = [int(i) for i in instances]
    
    return instances

def print_configuration(instance, model, method): 
    print("\n\n***********************************************************************")
    print("******Multiple Currier Capacitated Vehicle Routing Problem Solver******")
    print("***********************************************************************")

    print(f"Solving instance: {instance}, model: {model}, method: {method}\n")
    print("***********************************************************************\n")

def get_cp_model_path(model_name):
    cp_model_path = os.path.join(BASE_PATH, "CP", "CP_Model", model_name)
    return cp_model_path

def get_mip_model_path(model_name):
    mip_model_path = os.path.join(BASE_PATH, "ILP", model_name)
    return mip_model_path

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

def check_model_folder_exists(method): 
    #check if the folder with name equal to the model exists in the folder res
    path = os.path.join(BASE_PATH, "res", method)
    if not os.path.isdir(path):
        os.makedirs(path)

def check_instance_file_exists(instance_n, method):
    #check if the instance file exists in the folder res
    path = os.path.join(BASE_PATH, "res", method, f"{instance_n}.json")
    if not os.path.isfile(path):
        with open(path , 'w') as file:
            file.write("{}")

## TO DO continue from here
def update_json_file(instance_n, method, model_name, result):
    path = os.path.join(BASE_PATH, "res", method, f"{instance_n}.json")
    #open the json and update it with the new result
    with open(path, 'r') as file:
        data = file.read()
        data = json.loads(data)
        data[model_name] = {}
        
        data[model_name]["time"] = math.floor(result["time"])
        data[model_name]["optimal"] = result["optimal"]
        data[model_name]["obj"] = int(result["obj"])
        data[model_name]["sol"] = result["sol"]

    with open(path, 'w') as file:
        json.dump(data, file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script that takes method, model, and instance as input.")
    parser.add_argument('--method',     type=str,  required=True,  default="",      help='The method to use')
    parser.add_argument('--model',      type=str,  required=True,  default="",      help='The model to use')
    parser.add_argument('--instance',   type=str,  required=True,  default="1",     help='The number of the instances to solve')
    parser.add_argument('--solver',     type=str,  required=False, default="",      help='The solver to use')
    parser.add_argument('--timeout',    type=int,  required=False, default=300,     help='The timeout expressed in seconds')
    parser.add_argument('--int_res',    type=bool, required=False, default=False,   help='If true shows intermediate results. Buggy feature.')
    parser.add_argument('--update_json',type=str,  required=False, default='y',     help='If y update the json with the result. If n the pogram not update the json.')
    parser.add_argument('--use_sb',     type=str,  required=False, default='y',     help='If y add the sb constraint in the cp model when the matrix is symmetric. Otherwise type n.') 

    args = parser.parse_args()
    model_name = args.model
    method = args.method
    solver_id = args.solver
    timeout_time = args.timeout
    instances = define_instances_num(args.instance)
    int_res = args.int_res
    update_json = args.update_json
    use_sb = args.use_sb
    
    if update_json == 'y':
        update_json = True
    elif update_json == 'n':
        update_json = False
    else:
        print("update_json parameter not recognized. Please type y or n.")
        exit()

    if use_sb == 'y':
        use_sb = True
    elif use_sb == 'n':
        use_sb = False
    else:
        print("use_sb parameter not recognized. Please type y or n.")
        exit()

    print_configuration(instances, args.model, args.method)

    #select the model path depending on the selected method

    for instance_n in instances: 

        print(f"INSTANCE: {instance_n}\n")

        instance_file_name = (f"inst{instance_n}" if instance_n >= 10 else f"inst0{instance_n}") + ".dat"
        instance_path = os.path.join(BASE_PATH, "Instances", instance_file_name)

        if not os.path.isfile(instance_path):
            print(f"The instance file can't be found. \nGiven instance_path: {instance_path}\nCheck that the file exists")
            exit()

        #extract the data from the dat file
        num_vehicles, num_clients, vehicles_capacity, packages_size, distances = extract_data_from_dat(instance_path)

        instance = {
            "num_vehicles": num_vehicles,
            "num_clients": num_clients,
            "vehicles_capacity": vehicles_capacity,
            "packages_size": packages_size,
            "distances": distances
        }

        if method == "CP":
            result = solvers.solve_cp(model_name, solver_id, instance, timeout_time, use_sb)
        elif method == "MIP":
            result = solvers.solve_mip(instance, timeout_time, model_name)
        elif method == "SMT":
            result = solvers.solve_smt(model_name, instance, instance_n, timeout_time)
        elif method == "SAT":
            result = solvers.solve_sat(instance, instance_n, timeout_time, model_name)
        else:
            print(f"Method {method} not recognized. Please select one of the following: CP, MIP, SMT, SAT")
            exit()

        #result: "time": 300, "optimal": false, "obj": 12, "sol" : [[3, 6, 5], [4, 2], [7, 1]]
    
        if update_json:
            check_model_folder_exists(method)
            check_instance_file_exists(instance_n, method)
            update_json_file(instance_n, method, model_name, result)