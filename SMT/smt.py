from z3 import *
"""
LEGGI IL PDF PER CAPIRE COME è MODELLATO IL PROBLEMA, APPROCCIO 2DIMENSIONALE

"""


# Definisco i parametri
n = 2  # numero di righe
m = 6  # numero di colonne

# Creo la matrice x di variabili intere
x = [[Int(f'x_{i}_{j}') for j in range(m+1)] for i in range(n)]

# Creo un solver Z3
solver = Solver()

# Aggiungo i vincoli:
# - Ogni x[i,j] è maggiore o uguale a 0 e minore o uguale a n+1
for i in range(n):
    for j in range(m+1):
        solver.add(x[i][j] >= 0, x[i][j] <= m + 1)

# Vincolo: esattamente un elemento diverso da zero per ogni colonna (cliente visitato da un solo corriere) tranne l'ultima
for j in range(m):
    solver.add(Sum([If(x[i][j] != 0, 1, 0) for i in range(n)]) == 1)

# Vincolo: diagonale = 0
for i in range(n):
    for j in range(m+1):
        solver.add(x[i][j] != j+1)

# Corriere parte da un deposito
for i in range(n):
    solver.add(Sum([If(x[i][j] == m, 1, 0) for j in range(m+1)]) == 1)

# Corriere arriva in un deposito
for i in range(n):
    solver.add(x[i][m] > 0)


# I due constraint qui sotto sono strani ma dovrebbero funzionare
# Colonne diverse tra di loro (ad eccezione dello zero)
for j in range(m+1):
    for i in range(n):
        for w in range(i+1, n):
            solver.add(Or(x[i][j] == 0, x[w][j] == 0, x[i][j] != x[w][j]))

# righe con elementi diversi (ammessi più zeri)
for i in range(n):
    for w in range(m+1):
        for j in range(w+1, m+1):
            solver.add(Or(x[i][w] == 0, x[i][j] == 0, x[i][w] != x[i][j]))

# Verifica se il modello è soddisfacibile
if solver.check() == sat:
    model = solver.model()
    for i in range(n):
        print([model[x[i][j]] for j in range(m+1)])
else:
    print("Il modello non è soddisfacibile")



"""
Quello che manca è:
    - Vincolo del path coerente
    - Pacchi consegnati a tutti i clienti
    - Minimizzare il percorso massimo
"""