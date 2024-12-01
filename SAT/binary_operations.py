from z3 import *

def is_binary_number_leq(v, u):
    """
    Returns a Z3 expression enforcing that the binary number v is less than or equal to u,
    where v and u can have different lengths.
    
    Args:
        v (list[Bool]): Binary representation of the first number (most significant bit first).
        u (list[Bool]): Binary representation of the second number (most significant bit first).
    
    Returns:
        Z3 expression: A logical expression that is True if and only if v <= u.
    """
    len_v, len_u = len(v), len(u)
    
    if len_v < len_u:
        # Pad v with leading zeros to match the length of u
        v = [False] * (len_u - len_v) + v
    elif len_v > len_u:
        # Pad u with leading zeros to match the length of v
        u = [False] * (len_v - len_u) + u
    
    # Recursive implementation for v <= u
    def leq_recursive(v, u, digits):
        if digits == 1:
            # Base case: one-bit comparison
            return Or(Not(v[0]), u[0])  # v[0] <= u[0]
        else:
            # Recursive case
            return Or(
                And(Not(v[0]), u[0]),                     # v is strictly less than u at the MSB
                And(v[0] == u[0], leq_recursive(v[1:], u[1:], digits - 1))  # MSBs are equal, recurse on remaining bits
            )
    
    return leq_recursive(v, u, len(v))  # len(v) == len(u) after padding

def sum_binary_numbers(v, u):
    """
    Returns a Z3 expression that computes the sum of two binary numbers v and u,
    represented as lists of Bool variables.
    
    Args:
        v (list[Bool]): Binary representation of the first number (most significant bit first).
        u (list[Bool]): Binary representation of the second number (most significant bit first).
    
    Returns:
        list[Bool]: A list of Bool variables representing the binary sum of v and u.
    """
    len_v, len_u = len(v), len(u)
    
    # Pad the shorter number with leading False (zeros) to align lengths
    if len_v < len_u:
        v = [False] * (len_u - len_v) + v
    elif len_v > len_u:
        u = [False] * (len_v - len_u) + u
    
    max_len = max(len_v, len_u)
    sum_result = []
    carry = False  # Initial carry is 0 (False)

    # Full adder logic for each bit from least significant to most significant
    for i in range(max_len - 1, -1, -1):
        # Compute the sum bit and carry
        sum_bit = Xor(Xor(v[i], u[i]), carry)  # v[i] ⊕ u[i] ⊕ carry
        carry = Or(And(v[i], u[i]), And(carry, Xor(v[i], u[i])))  # Carry out
        
        # Append the result bit to the front
        sum_result.insert(0, sum_bit)
    
    # Handle the final carry (overflow)
    sum_result.insert(0, carry)

    return sum_result

def sum_binary_numbers(v, u):
    """
    Returns a Z3 expression that computes the sum of two binary numbers v and u,
    represented as lists of Bool variables.
    
    Args:
        v (list[Bool]): Binary representation of the first number (most significant bit first).
        u (list[Bool]): Binary representation of the second number (most significant bit first).
    
    Returns:
        list[Bool]: A list of Bool variables representing the binary sum of v and u.
    """
    len_v, len_u = len(v), len(u)
    
    # Pad the shorter number with leading False (zeros) to align lengths
    if len_v < len_u:
        v = [False] * (len_u - len_v) + v
    elif len_v > len_u:
        u = [False] * (len_v - len_u) + u
    
    max_len = max(len_v, len_u)
    sum_result = []
    carry = False  # Initial carry is 0 (False)

    # Full adder logic for each bit from least significant to most significant
    for i in range(max_len - 1, -1, -1):
        # Compute the sum bit and carry
        sum_bit = Xor(Xor(v[i], u[i]), carry)  # v[i] ⊕ u[i] ⊕ carry
        carry = Or(And(v[i], u[i]), And(carry, Xor(v[i], u[i])))  # Carry out
        
        # Append the result bit to the front
        sum_result.insert(0, sum_bit)
    
    # Handle the final carry (overflow)
    sum_result.insert(0, carry)
    print(sum_result[0])

    return sum_result

# Example usage
v = [Bool(f"v_{i}") for i in range(3)]  # 3-bit binary number
u = [Bool(f"u_{i}") for i in range(4)]  # 4-bit binary number

# Create the sum
sum_result = sum_binary_numbers(v, u)

# Solver
solver = Solver()

# Add constraints for testing
solver.add(v[0] == False, v[1] == True, v[2] == True)  # v = 3 (011 in 3 bits)
solver.add(u[0] == True, u[1] == True, u[2] == False, u[3] == True)  # u = 13 (1101 in 4 bits)

# Add constraints to verify sum
solver.add(sum_result[0] == True,  # No overflow
           sum_result[1] == False,
           sum_result[2] == False,
           sum_result[3] == False,
           sum_result[4] == False)  # Result should be 16 (10000)

# Check satisfiability
if solver.check() == sat:
    model = solver.model()
    print("Satisfiable assignment:")
    for b in v + u + sum_result:
        print(f"{b} = {model[b]}")
else:
    print("Unsatisfiable")