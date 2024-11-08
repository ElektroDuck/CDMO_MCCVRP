import os
import argparse
import solvers


BASE_PATH = os.getcwd()


def define_instances_num(instances): 
    if instances == "all":
        instances =  [i for i in range(1,22)] 
    elif "-" in instances:
        limits = instances.split("-")
        print(limits, int(limits[0]), int(limits[1])+1)
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


if __name__ == "__main__":
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
            result = solvers.solve_cp(model_name, solver_id, instance, timeout_time, int_res)
        elif method == "MIP":
            result = solvers.solve_mip(instance, timeout_time)
        elif method == "SMT":
            result = solvers.solve_smt(instance, timeout_time)