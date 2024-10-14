import os
import argparse
from minizinc import Instance, Model, Solver

#python run_model.py --method cp --model Model_A.mzn --instance 1 --solver chuffed

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

def main():
    parser = argparse.ArgumentParser(description="Script that takes method, model, and instance as input.")
    parser.add_argument('--method', type=str, required=True, help='The method to use')
    parser.add_argument('--model', type=str, required=True, help='The model to use')
    parser.add_argument('--instance', type=int, required=True, help='The instance to use')
    parser.add_argument('--solver', type=str, required=True, help='The solver to use')

    args = parser.parse_args()
    model_name = args.model
    instance_num = args.instance
    method = args.method
    solver_id = args.solver


    print_configuration(args.instance, args.model, args.method)

    base_path = os.getcwd()
    model_path = os.path.join(base_path, "CP", "CP_Model", model_name)


    instance_file_name = (f"inst{instance_num}" if instance_num >= 10 else f"inst0{instance_num}") + ".dat"
    instance_path = os.path.join(base_path, "Instances", instance_file_name)

    #check if the files exists
    if not os.path.isfile(model_path):
        print(f"The model file can't be found. \nGiven model_ path: {model_path}\nCheck that the file exists")
        exit()
    if not os.path.isfile(instance_path):
        print(f"The instance file can't be found. \nGiven instance_path: {instance_path}\nCheck that the file exists")
        exit()

    #extract the data from the dat file
    num_vehicles, num_clients, vehicles_capacity, packages_size, distances = extract_data_from_dat(instance_path)

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
    result = instance.solve()

    #print the result
    print(result)

if __name__ == "__main__":
    main()