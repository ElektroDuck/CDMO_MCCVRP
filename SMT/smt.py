from z3 import *
"""
LEGGI IL PDF PER CAPIRE COME è MODELLATO IL PROBLEMA, APPROCCIO 2DIMENSIONALE

"""


# Definisco i parametri
n = 6  # numero di righe
m = 2  # numero di colonne

# Creo la matrice x di variabili intere
x = [[Int(f'x_{i}_{j}') for j in range(m)] for i in range(n+1)]

# Creo un solver Z3
solver = Solver()

# Aggiungo i vincoli:
# - Ogni x[i,j] è maggiore o uguale a 0 e minore o uguale a n+1
for i in range(n):
    for j in range(m):
        solver.add(x[i][j] >= 0, x[i][j] <= n + 1)

# Vincolo: esattamente un elemento diverso da zero per ogni riga (cliente visitato da un solo corriere)
for i in range(n):
    solver.add(Sum([If(x[i][j] != 0, 1, 0) for j in range(m)]) == 1)

# Vincolo: diagonale = 0
for j in range(m):
    for i in range(n+1):
        solver.add(x[i][j] != i+1)

# Corriere parte da un deposito
for j in range(m):
    solver.add(x[n][j] > 0)

# Corriere arriva in un deposito
for j in range(m):
    solver.add(Sum([If(x[i][j] == n+1, 1, 0) for i in range(n)]) == 1)


# Verifica se il modello è soddisfacibile
if solver.check() == sat:
    model = solver.model()
    for i in range(n+1):
        print([model[x[i][j]] for j in range(m)])
else:
    print("Il modello non è soddisfacibile")



"""
Quello che manca è:
    - Vincolo del path coerente
    - Pacchi consegnati a tutti i clienti
    - Minimizzare il percorso massimo
"""