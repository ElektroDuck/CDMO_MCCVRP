from minizinc import Instance, Model, Solver

file = "C:\\Users\\lucab\\Desktop\\universitá\\CDMO\\CDMO_MCCVRP\\python\\nqueens.mzn"

# Load n-Queens model from file
nqueens = Model(file)
# Find the MiniZinc solver configuration for Gecode
gecode = Solver.lookup("gecode")
# Create an Instance of the n-Queens model for Gecode
instance = Instance(gecode, nqueens)
# Assign 4 to n
instance["n"] = 4
result = instance.solve()
# Output the array q
print(result["q"])