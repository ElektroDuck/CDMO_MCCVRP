from z3 import * 

from utils import *
from convertion_functions import *
from binary_operations import *


def obj_function(model, distances):
    """Given a model, returns the objective function value that we are interested in (i.e. max of distances) as an integer

    Args:
        model (ModelRef): model of which to compute the objective function
        distances (list[list[Bool]]): list containing the binary representation (using Z3 Bool variables) of each distance

    Returns:
        int: the maximum distance travelled
    """
    m = len(distances)
    maxdist = -1
    for i in range(m):
        dist = bin_to_int([
            1 if model.evaluate(distances[i][j]) else 0
            for j in range(len(distances[i]))
        ])
        maxdist = max(maxdist, dist)
    return maxdist


def AllLessEq_bin(distances, upper_bound_bin):
    """Encodes the constraint {Forall i. distances[i] <= upper_bound_bin}

    Args:
        distances (list[list[Bool]]): list containing the binary representation (using Z3 Bool variables) of each distance
        upper_bound_bin (list[bool]): binary representation of the upper bound

    Returns:
        Z3-Expression: the constraint encoding
    """
    m = len(distances)

    clauses = []

    for i in range(m):
        clauses.append(is_binary_number_leq(distances[i], upper_bound_bin))

    return And(clauses)


def AtLeastOneGreaterEq_bin(distances, lower_bound_bin):
    """Encodes the constraint {Exists i. distances[i] >= lower_bound_bin}

    Args:
        distances (list[list[Bool]]): list containing the binary representation (using Z3 Bool variables) of each distance
        lower_bound_bin (list[bool]): binary representation of the lower bound

    Returns:
        Z3-Expression: the constraint encoding
    """
    m = len(distances)

    clauses = []

    for i in range(m):
        clauses.append(is_binary_number_leq(lower_bound_bin, distances[i]))

    return Or(clauses)