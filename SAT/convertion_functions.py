import math
from z3 import BitVec

def num_bits(value):
    """
    Computes the number of bits needed to store a given positive integer value.
    
    Parameters:
    value (int): The positive integer value to compute the bit length for.
    
    Returns:
    int: The number of bits required to store the value.
    """
    if value == 0:
        return 1  # 0 needs 1 bit
    return math.floor(math.log2(value)) + 1

def convert_int_bits(value, num_bits, string_var="Value"):
    """
    Converts a positive integer into a binary Z3 BitVec element.
    
    Parameters:
    value (int): The positive integer value to convert.
    num_bits (int): The number of bits needed to represent the value.
    string_var (str): The name associated with the variable

    Returns:
    BitVec: The Z3 BitVec representation of the value.
    """
    if value < 0:
        raise ValueError("Value must be positive.")
    if num_bits < 1:
        raise ValueError("Number of bits must be at least 1.")
    
    # Create a Z3 BitVec element with the specified number of bits
    return BitVec(string_var+f"_{value}", num_bits)

def bitvec_to_int(bitvec, model):
    """
    Converts a Z3 BitVec value to an integer using the provided model.
    
    Args:
        bitvec (BitVec): The BitVec value from a Z3 solver.
        model (ModelRef): The Z3 model containing the solution.

    Returns:
        int: The integer value of the BitVec.
    """
    return model.eval(bitvec).as_long()