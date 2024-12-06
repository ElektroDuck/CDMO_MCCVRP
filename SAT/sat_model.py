import time
from z3 import *


import json
import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from convertion_functions import *
from binary_logic import *
from binary_operations import *
from checks_obj_func import *
from utils import *
from cycles import *
from show import *

def creat_solution_dict(paths, time, optimal, obj_value):
    return {
        "time" : time if time <= 300 else 300,
        "optimal" : optimal,
        "obj" : obj_value,
        "sol" : paths
    }

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

def sat_model(num_couriers, num_items, capacity, size, distances_int, upper_bound, lower_bound, display_solution=True, timeout_duration=300, search="Base"):
    ### VARIABLES

    nodes= num_items+1 #considering the depot as the n+1 location
    # assignments courier-item
    assignments = [[Bool(f"assignment_{i}_{j}") for j in range(num_items)] for i in range(num_couriers)]
    # assignment_ij = 1 indicates that courier i delivers object j

    # routes
    routes = [[[Bool(f"route_{i}_{j}_{k}") for k in range(nodes)] for j in range(nodes)] for i in range(num_couriers)]
    # route_ijk = 1 indicates that courier i moves from delivery point j to delivery point k in his route
    
    # order
    order = [[Bool(f"deliver_{j}_{k}-th") for k in range(num_items)] for j in range(num_items)]
    # order_jk == 1 iff object j is delivered as k-th in its courier's route

    #representation of actual load carried by the couriers
    courier_loads = [[Bool(f"cl_{i}_{k}") for k in range(num_bits(max(capacity)))] for i in range(num_couriers)] 
    # courier_loads_i = binary representation of actual load carried by each courier

    ### CREATE SOLVER
    solver = z3.Solver()

    ### CONSTRAINT

    #fare conversione di capacity e size in bit
    cap_bin = [int_to_bin(capacity_i, num_bits(capacity_i)) for capacity_i in capacity]
    size_bin = [int_to_bin(size_i, num_bits(size_i)) for size_i in size]

    # Every object is assigned to one and only one courier
    for j in range(num_items):
        solver.add(exactly_one([assignments[i][j] for i in range(num_couriers)]))

    # Every courier can't exceed its load capacity
    for i in range(num_couriers):
        solver.add(sum_K_bin(assignments[i], size_bin, courier_loads[i], f"courier_loads_computed_{i}"))
        solver.add(is_binary_number_leq(courier_loads[i], cap_bin[i]))

    # Every courier leaves the depot (implied constraint, because n >= m)
    for i in range(num_couriers):
        solver.add(at_least_one(assignments[i]))

    # Every object is delivered only one time in the order
    for i in range(num_items):
        solver.add(exactly_one(order[i]))

    for i in range(num_couriers):
        # the couriers can't leave from j to go to j
        solver.add(And([Not(routes[i][j][j]) for j in range(num_items)]))
        
        solver.add(Not(routes[i][num_items][num_items]))     # don't let courier i have a self loop

        # Row j has a 1 if courier i delivers object j
        for j in range(num_items):
            solver.add(Implies(assignments[i][j], exactly_one(routes[i][j]))) # If assignments_ij then exactly_one(routes_ij)
            solver.add(Implies(Not(assignments[i][j]), all_false(routes[i][j]))) # else all_false(routes_ij)
        solver.add(exactly_one(routes[i][nodes-1])) # exactly_one in origin point row === courier i leaves from origin

        # Column j has a 1 if courier i delivers object j
        # columns
        for k in range(num_items):
            solver.add(Implies(assignments[i][k], exactly_one([routes[i][j][k] for j in range(nodes)]))) # If assigments_ij then exactly_one(routes_i,:,k)
           #solver.add(Implies(Not(assignments[i][k]), all_false([routes[i][j][k] for j in range(nodes)]))) # else all_false(routes_i,:,k)
        solver.add(exactly_one([routes[i][j][nodes-1] for j in range(nodes)])) # exactly_one in origin point column === courier i returns to origin

        # Avoid loops without the origin
        for j in range(num_items):
            for k in range(num_items):
                solver.add(Implies(routes[i][j][k], link_true_indices(order[j], order[k])))
            solver.add(Implies(routes[i][nodes-1][j], order[j][0]))

    distances = [[Bool(f"dist_bin_{i}_{k}") for k in range(num_bits(upper_bound))] for i in range(num_couriers)]

    # flatten r and D
    flat_r = [flatten(routes[i]) for i in range(num_couriers)]
    flat_d = flatten(distances_int)
    # convert flat_D to binary
    flat_d_bin = [int_to_bin(e, num_bits(e) if e > 0 else 1) for e in flat_d]

    #Distances travelled by each courier

    distances = [[Bool(f"dist_bin_{i}_{k}") for k in range(num_bits(upper_bound))] for i in range(num_couriers)]

    # Definition of distances
    for i in range(num_couriers):
        solver.add(sum_K_bin(flat_r[i], flat_d_bin, distances[i], f"distances_computed_{i}"))

    ### SEARCH OF OPTIMAL SOLUTION

    model = None
    obj_value = None
    encoding_time = time.time()

    timeout = encoding_time + timeout_duration

    solver.push()

    solv_res = solver.check()

    if search == "Base":

        solver.set('timeout', millisecs_left(time.time(), timeout))
        while solver.check() == z3.sat:

            model = solver.model()
            obj_value = obj_function(model, distances)
            print(f"This model obtained objective value: {obj_value} after {round(time.time() - encoding_time, 1)}s")

            if obj_value <= lower_bound:
                break

            upper_bound = obj_value - 1
            upper_bound_bin = int_to_bin(upper_bound, num_bits(upper_bound))

            solver.pop()
            solver.push()

            solver.add(AllLessEq_bin(distances, upper_bound_bin))
            now = time.time()
            if now >= timeout:
                break
            solver.set('timeout', millisecs_left(now, timeout))

    elif search == "Binary":
        
        upper_bound_bin = int_to_bin(upper_bound, num_bits(upper_bound))
        print(type(distances[0]))
        print(type(upper_bound_bin[0]))
        for i in range(len(upper_bound_bin)):
            upper_bound_bin[i] = bool(upper_bound_bin)
        solver.add(AllLessEq_bin(distances, upper_bound_bin))

        lower_bound_bin = int_to_bin(lower_bound, num_bits(lower_bound))
        for i in range(len(lower_bound_bin)):
            lower_bound_bin[i] = bool(lower_bound_bin)
        solver.add(AtLeastOneGreaterEq_bin(distances, lower_bound_bin))

        while lower_bound <= upper_bound:
            mid = int((lower_bound + upper_bound)/2)
            mid_bin = int_to_bin(mid, num_bits(mid))
            solver.add(AllLessEq_bin(distances, mid_bin))

            now = time.time()
            if now >= timeout:
                break
            solver.set('timeout', millisecs_left(now, timeout))
            print(f"Trying with bounds: [{lower_bound}, {upper_bound}] and posing obj_val <= {mid}")

            if solver.check() == z3.sat:
                model = solver.model()
                obj_value = obj_function(model, distances)
                print(f"This model obtained objective value: {obj_value} after {round(time.time() - encoding_time, 1)}s")

                if obj_value <= 1:
                    break

                upper_bound = obj_value - 1
                upper_bound_bin = int_to_bin(upper_bound, num_bits(upper_bound))


            else:
                print(f"This model failed after {round(time.time() - encoding_time, 1)}s")

                lower_bound = mid + 1
                lower_bound_bin = int_to_bin(lower_bound, num_bits(lower_bound))

            solver.pop()
            solver.push()
            solver.add(AllLessEq_bin(distances, upper_bound_bin))
            solver.add(AtLeastOneGreaterEq_bin(distances, lower_bound_bin))

    else:
        raise ValueError(f"Input parameter [search] mush be either 'Linear' or 'Binary', was given '{search}'")

    # compute time taken
    end_time = time.time()
    if end_time >= timeout:
        solving_time = timeout_duration    # solving_time has upper bound of timeout_duration if it timeouts
    else:
        solving_time = math.floor(end_time - encoding_time)

    # if no model is found -> UNSAT if solved to optimality else UNKKNOWN
    if model is None:
        ans = "N/A" if solving_time == timeout_duration else "UNSAT"
        return (ans, solving_time, None)

    # check that all couriers travel hamiltonian cycles
    R = evaluate(model, routes)
    assert(check_all_hamiltonian(R))

    Ord = evaluate(model, order)
    Ass = evaluate(model, assignments)
    
    if display_solution:
        Dists = evaluate(model, distances)
        showMCCVRP(Ord, Dists, obj_value, Ass)

    deliveries = retrieve_routes(Ord, Ass)

    
    # Run the solver
    if solv_res == sat:
        res_dict = creat_solution_dict(deliveries, solving_time, True, obj_value)
        with open("tmp_output.json", "w") as f:
            json.dump(res_dict, f)

    # If the model is not sat...
    else:
        result_dict = {
        "time" : 300,
        "optimal" : False,
        "obj" : 0,
        "sol" : []
        }
        with open("tmp_output.json", "w") as f:
            json.dump(result_dict, f)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inst")
    parser.add_argument("ub")
    parser.add_argument("lb")
    args = parser.parse_args()

    instance_number = int(args.inst)
    ub = int(args.ub)
    lb = int(args.lb)

    num_vehicles, num_clients, vehicles_capacity, packages_size, distances = extract_data_from_dat(instance_number)
    obj_value, solving_time, deliveries = sat_model(num_vehicles, num_clients, vehicles_capacity, packages_size, distances, ub, lb)
    #TODO: salvare su file questo output