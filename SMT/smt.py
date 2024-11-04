from z3 import *
import os
import time
"""
LEGGI IL PDF PER CAPIRE COME è MODELLATO IL PROBLEMA, APPROCCIO 2DIMENSIONALE
"""


def solve(instance):
    if os.path.exists("../Instances"):
        with open("../Instances/"+instance, 'r') as file:
            lines = file.readlines()
        
        num_vehicles = int(lines[0].strip())
        num_clients = int(lines[1].strip())
        
        vehicles_capacity = list(map(int, lines[2].strip().split()))
        packages_size = list(map(int, lines[3].strip().split()))

        distances = [list(map(int, line.strip().split())) for line in lines[4:]]





    # Definisco i parametri
    n = num_vehicles  # numero di righe
    m = num_clients  # numero di colonne

    # Creo la matrice x di variabili intere
    x = [[Int(f'x_{i}_{j}') for j in range(m+1)] for i in range(n)]
    total_distances = [Int(f"total_distance_{i}") for i in range(n)]
    max_distance = Int("max_distance")
    #u = [[Int(f'u_{i}_{j}') for j in range(m+1)] for i in range(n)]


    # Creo un solver Z3
    solver = Optimize()

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
        solver.add(Sum([If(x[i][j] == m+1, 1, 0) for j in range(m+1)]) == 1)

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


    # Constraint della capacità
    for i in range(n):
        #vehicles_capacity
            solver.add(Sum([If(x[i][j] != 0, packages_size[j], 0) for j in range(m)]) <= vehicles_capacity[i])
    


    # Funzione obiettivo
    for i in range(n):
        solver.add(total_distances[i] == Sum([
            Sum([
                If(x[i][j] == k + 1, distances[k][j], 0)  # Se x[i][j] è la destinazione k+1, usa distances[j][k]
                for k in range(m+1)
            ])
            for j in range(m+1)
        ]))

    for i in range(n):
        solver.add(max_distance >= total_distances[i])

    # Verifica se il modello è soddisfacibile
    # if solver.check() == sat:
    #     model = solver.model()
    #     for i in range(n):    
    #         print([model[x[i][j]] for j in range(m+1)])
    # else:
    #     print("Il modello non è soddisfacibile")

    solver.minimize(max_distance)
    if solver.check() == sat:
        
        model = solver.model()
        #print("Distanza massima minimizzata:", model[max_distance])
        return True, model[max_distance]
        for i in range(n):
            print(f"Distanza totale per corriere {i}: ", model[total_distances[i]])
            print(f"Percorso del corriere {i}: ", [model[x[i][j]] for j in range(m+1)])
            print("\n")
    else:
        return False, None
        print("Nessuna soluzione trovata.")


for i in range(1,22):
    instance = f"inst0{i}.dat" if i < 10 else f"inst{i}.dat"
    start = time.time()
    result, max_dist = solve(instance)
    end = time.time()
    if result:
        print(f"Instance {i} solved in {round(end-start, 2)}s with a max_dist of {max_dist}")
    else:
        print(f"instance {i} not solved (took {round(end-start, 2)}s)")




"""
Quello che manca è:
    - Vincolo del path coerente
    - Pacco consegnato da un solo corriere (in teoria minimizzando poi il percorso massimo questa roba dovrebbe essere implicita)
"""