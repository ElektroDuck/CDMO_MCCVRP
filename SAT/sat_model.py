from z3 import *
from convertion_functions import *
from binary_logic import *

def sat_model(num_couriers, num_items, capacity, size, distances, upper_bound, lower_bound, symmetric=True, display_solution=True, timeout_duration=300):
    ### VARIABLES

    nodes= num_items+1 #considering the depot as the n+1 location
    # assignments courier-item
    assignments = [[Bool(f"a_{i}_{j}") for j in range(num_items)] for i in range(num_couriers)]
    # a_ij = 1 indicates that courier i delivers object j

    # routes
    routes = [[[Bool(f"r_{i}_{j}_{k}") for k in range(nodes)] for j in range(nodes)] for i in range(num_couriers)]
    # r_ijk = 1 indicates that courier i moves from delivery point j to delivery point k in his route
    
    # order
    order = [[Bool(f"deliver_{j}_as_{k}-th") for k in range(num_items)] for j in range(num_items)]
    # t_jk == 1 iff object j is delivered as k-th in its courier's route (intuition of time)

    #representation of actual load carried by the couriers
    courier_loads = [[Bool(f"cl_{i}_{k}") for k in range(convert_int_bits(sum(size)))] for i in range(num_couriers)] #fare metodo conversione int->bits
    # courier_loads_i = binary representation of actual load carried by each courier

    ### CREATE SOLVER
    solver = z3.Solver()

    ### CONSTRAINT

    #fare conversione di capacity e size in bit
    cap_bit = convert_int_bits(capacity, num_bits(capacity))
    size_bit = convert_int_bits(size, num_bits(size))

    # Every object is assigned to one and only one courier
    for j in range(num_items):
        solver.add(exactly_one_seq([assignments[i][j] for i in range(num_couriers)]) #fare metodo exactly one per righe

    # Every courier can't exceed its load capacity
    for i in range(num_couriers):
        solver.add(sum_K_bin(assignments[i], s_bin, courier_loads[i])) #fare metodo somma
        solver.add(leq(courier_loads[i], l_bin[i]))

    # Every courier leaves the depot (implied constraint, because n >= m)
    for i in range(num_couriers):
        solver.add(at_least_one(assignments[i]))

    # Every object is delivered only one time in the order
    for i in range(num_items):
        solver.add(exactly_one_seq(order[i]))

    # Routes
    for k in range(num_couriers):
        for i in range(nodes):
            solver.add(exaclty_one_seq(routes[i][j][k] for j in range(nodes)))

    for i in range(nodes):
        for j in range(nodes):
            solver.add(exaclty_one_seq(routes[i][j][k] for k in range(num_couriers)))

    for k in range(num_couriers):
        for j in range(nodes):
            solver.add(exaclty_one_seq(routes[i][j][k] for i in range(nodes)))

    for i in range(num_couriers):
        # the couriers can't leave from j to go to j
        solver.add(And([Not(r[i][j][j]) for j in range(num_items)]))
        
        solver.add(Not(routes[i][n][n]))     # don't let courier i have a self loop

        # Row j has a 1 if courier i delivers object j
        for j in range(num_items):
            solver.add(Implies(assignments[i][j], exactly_one_seq(routes[i][j]))) # If assignments_ij then exactly_one(routes_ij)
            solver.add(Implies(Not(assignment[i][j]), all_false(routes[i][j]))) # else all_false(routes_ij)
        solver.add(exactly_one_seq(routes[i][nodes-1])) # exactly_one in origin point row === courier i leaves from origin

        # Column j has a 1 if courier i delivers object j
        # columns
        for k in range(num_items):
            solver.add(Implies(assignments[i][k], exactly_one_seq([routes[i][j][k] for j in range(nodes)]))) # If assigments_ij then exactly_one(routes_i,:,k)
            solver.add(Implies(Not(assigments[i][k]), all_false([routes[i][j][k] for j in range(nodes)]))) # else all_false(routes_i,:,k)
        solver.add(exactly_one_seq([routes[i][j][nodes-1] for j in range(nodes)])) # exactly_one in origin point column === courier i returns to origin

        # Avoid loops without the origin
        for j in range(num_items):
            for k in range(num_items):
                solver.add(Implies(routes[i][j][k], successive(order[j], order[k])))
            solver.add(Implies(routes[i][nodes-1][j], order[j][0]))


    max_distances = [max(distances[i][:-1]) for i in range(num_items)]
    distances = [[Bool(f"dist_bin_{i}_{k}") for k in range(num_bits(upper_bound))] for i in range(num_couriers)]

    # Definition of distances
    for i in range(num_couriers):
        solver.add(sum_K_bin(flat_r[i], flat_D_bin, distances[i]))

    ### SEARCH OF OPTIMAL SOLUTION

    model = None
    obj_value = None
    
    solver.push()

    solver-check()

    #####CONTROLLA DA QUI##########

#    solver.set('timeout', millisecs_left(time.time(), timeout))
#    while solver.check() == z3.sat:
#
#        model = solver.model()
#        obj_value = obj_function(model, distances)
#        print(f"This model obtained objective value: {obj_value} after {round(time.time() - encoding_time, 1)}s")
#
#        if obj_value <= lower_bound:
#            break
#
#        upper_bound = obj_value - 1
#        upper_bound_bin = int_to_bin(upper_bound, num_bits(upper_bound))
#
#        solver.pop()
#        solver.push()
#
#        solver.add(AllLessEq_bin(distances, upper_bound_bin))
#        now = time.time()
#        if now >= timeout:
#            break
#        solver.set('timeout', millisecs_left(now, timeout))
#
#     # compute time taken
#    end_time = time.time()
#    if end_time >= timeout:
#        solving_time = timeout_duration    # solving_time has upper bound of timeout_duration if it timeouts
#    else:
#        solving_time = math.floor(end_time - encoding_time)
#
#    # if no model is found -> UNSAT if solved to optimality else UNKKNOWN
#    if model is None:
#        ans = "N/A" if solving_time == timeout_duration else "UNSAT"
#        return (ans, solving_time, None)
#
#    # reorder all variables w.r.t. the original permutation of load capacities, i.e. of couriers
#    if symmetry_breaking:
#        a_copy = copy.deepcopy(a)
#        r_copy = copy.deepcopy(r)
#        for i in range(m):
#            a[permutation[i]] = a_copy[i]
#            r[permutation[i]] = r_copy[i]
#
#    # check that all couriers travel hamiltonian cycles
#    R = evaluate(model, r)
#    assert(check_all_hamiltonian(R))
#
#    T = evaluate(model, t)
#    A = evaluate(model, a)
#    
#    if display_solution:
#        Dists = evaluate(model, distances)
#        displayMCP(T, Dists, obj_value, A)
#
#    deliveries = retrieve_routes(T, A)
#
#    return (obj_value, solving_time, deliveries)
