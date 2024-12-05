from z3 import *
import os
import time
import argparse
from sat_model import *
import numpy as np


#Initialization of the instance
start = time.time()
parser = argparse.ArgumentParser()
parser.add_argument("inst")
args = parser.parse_args()

# Get the instance to read
instance_number = int(args.inst)

#Read the instance utility function
def extract_data_from_dat(instance_number):
    instance_number = str(instance_number) if instance_number >= 10 else "0"+str(instance_number)
    instance_path = f"{os.getcwd()}/Instances/inst{instance_number}.dat"
    with open(instance_path, 'r') as file:
        lines = file.readlines()
    
    num_vehicles = int(lines[0].strip())
    num_clients = int(lines[1].strip())
    
    vehicles_capacity = list(map(int, lines[2].strip().split()))
    packages_size = list(map(int, lines[3].strip().split()))

    distances = [list(map(int, line.strip().split())) for line in lines[4:]]

    return num_vehicles, num_clients, vehicles_capacity, packages_size, distances

def compute_bounds(distances, num_vehicles, num_clients):
    matrix_dist = np.array(distances) #transform in numpy matrix
    last_row = matrix_dist[-1, :]  # selects the last row
    last_column = matrix_dist[:, -1]  # selects the last column
    result = last_row + last_column
    low_bound = max(result)

    dist_sorted = matrix_dist[np.max(matrix_dist, axis=0).argsort()]
    up_bound = sum([max(dist_sorted[i]) for i in range(num_vehicles-1, num_clients+1)])

    return low_bound, up_bound

num_vehicles, num_clients, vehicles_capacity, packages_size, distances = extract_data_from_dat(instance_number)
low_bound, up_bound = compute_bounds(distances, num_vehicles, num_clients)

sat_model = sat_model(num_vehicles, num_clients, vehicles_capacity, packages_size, distances, up_bound, low_bound, display_solution=True, timeout_duration=300, search = "Base")