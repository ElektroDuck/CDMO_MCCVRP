import os
import argparse
from minizinc import Instance, Model, Solver

#python run_model.py --method method1 --model model_A.mzn --solver gecode --instance 1 

CP_MODELS_FOLDER = ".\CP\CP_Model\\"
INSTANCE_FOLDER = ".\Instances\\"

def print_configuration(instance, model, method): 
    print(f"solving instance: {instance}, model: {model}, method: {method}")

def extract_data_from_dat(instance_path, verbouse=False): 
    with open(instance_path, 'r') as file:
        lines = file.readlines()
    
    num_vehicles = int(lines[0].strip())
    num_clients = int(lines[1].strip())
    
    vehicles_capacity = list(map(int, lines[2].strip().split()))
    packages_size = list(map(int, lines[3].strip().split()))

    distances = [list(map(int, line.strip().split())) for line in lines[4:]]


    if verbouse: 
        print(f"num_vehicles: {num_vehicles}")
        print(f"num_clients: {num_clients}")
        print(f"packages_size:\n {packages_size}")
        print(f"vehicles_capacity:\n {vehicles_capacity}")
    
        print(f"distances:")
        for i in distances: 
            print(i)
    
    return num_vehicles, num_clients, vehicles_capacity, packages_size, distances

def main():
    parser = argparse.ArgumentParser(description="Script that takes method, model, and instance as input.")
    parser.add_argument('--method', type=str, required=True, help='The method to use')
    parser.add_argument('--model', type=str, required=True, help='The model to use')
    parser.add_argument('--instance', type=int, required=True, help='The instance to use')
    parser.add_argument('--solver', type=str, required=True, help='The solver to use', default='gecode')

    args = parser.parse_args()
    model_name = args.model
    instance_num = args.instance
    method = args.method
    solver = args.solver

    print_configuration(args.instance, args.model, args.method)

    model_path = CP_MODELS_FOLDER + model_name
    instance_path = INSTANCE_FOLDER + (f"inst{instance_num}" if instance_num >= 10 else f"inst0{instance_num}") + ".dat"
    
    #check if the files exists
    if not os.path.isfile(model_path):
        print(f"The model file can't be found. \nGiven model_ path: {model_path}\nCheck that the file exists")
        exit()
    if not os.path.isfile(instance_path):
        print(f"The instance file can't be found. \nGiven instance_path: {instance_path}\nCheck that the file exists")
        exit()

    num_vehicles, num_clients, vehicles_capacity, packages_size, distances = extract_data_from_dat(instance_path)

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

if __name__ == "__main__":
    main()