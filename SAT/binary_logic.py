from z3 import *

def at_least_one(bools):
    """
    Returns the Z3 expression enforcing that at least one of the given boolean variables is True.
    
    :param bools: List of Z3 boolean variables.
    :return: Z3 expression for "at least one".
    """
    return Or(bools)

def at_most_one(bools):
    """
    Returns the Z3 expression enforcing that at most one of the given boolean variables is True.
    
    :param bools: List of Z3 boolean variables.
    :return: Z3 expression for "at most one".
    """
    return And([Or(Not(bools[i]), Not(bools[j])) 
                for i in range(len(bools)) 
                for j in range(i + 1, len(bools))])

def exactly_one(bools):
    """
    Returns the Z3 expression enforcing that exactly one of the given boolean variables is True.
    
    :param bools: List of Z3 boolean variables.
    :return: Z3 expression for "exactly one".
    """
    return And(at_least_one(bools), at_most_one(bools))

def are_equal(list1, list2):
    """
    Returns a Z3 expression enforcing that two lists of boolean variables are equal.
    
    :param list1: First list of Z3 boolean variables.
    :param list2: Second list of Z3 boolean variables.
    :return: Z3 expression enforcing the equality of the two lists.
    """
    if len(list1) != len(list2):
        raise ValueError("Lists must be of the same length to compare equality.")
    
    # Pairwise equivalence between corresponding elements
    return And([list1[i] == list2[i] for i in range(len(list1))])

def all_false(bools):
    """
    Returns a Z3 expression enforcing that all boolean variables in the list are False.
    
    :param bools: List of Z3 boolean variables.
    :return: Z3 expression enforcing that all variables are False.
    """
    return And([Not(b) for b in bools])

def link_true_indices(l1, l2):
    """
    Returns a Z3 expression enforcing the constraint that the only true value in l2
    corresponds to the index i+1 of the true value in l1.
    
    :param l1: List of Z3 boolean variables for the first list.
    :param l2: List of Z3 boolean variables for the second list.
    :return: Z3 expression enforcing the described constraint.
    """
    if len(l2) != len(l1):
        raise ValueError("The length of l2 must be equal to the length of l1.")
    
    # Enforce l2[i+1] is `True` where l1[i] is `True`
    link_constraint = And([
        Implies(l1[i], l2[i + 1]) for i in range(len(l1)-1)
    ])
    constraint = And(link_constraint, Not(l1[-1]), Not(l2[0]))
    
    return constraint