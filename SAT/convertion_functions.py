import math
from z3 import *

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


def int_to_bin(value, bits):
    """Converts a positive integer into a binary representation of True/False using #bits = bits

    Args:
        value (int): the integer to convert
        bits (int): the number of bits of the output

    Returns:
        list[bool]: the binary representation of the integer
    """
    return [(value%(2**(i+1)) // 2**i)==1 for i in range(bits-1,-1,-1)]

def bin_to_int(value):
    """
    Converts a binary number of 1s and 0s into its integer form

    Parameters:
        value (list[int]): the binary number to convert

    Returns:
        int: the converted integer representation of x
    """
    n = len(value)
    x = sum(2**(n-i-1) for i in range(n) if value[i] == 1)
    return x