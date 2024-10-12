int: num_vehicles;
int: num_clients;

array[1..num_vehicles] of int: capacity;
array[1..num_clients] of int: size;
array[1..num_clients+1, 1..num_clients+1] of int: distances;                         
array[1..num_vehicles] of var int: theta;  % Ogni veicolo ha un proprio theta
var int: max_theta;  % Variabile per il massimo percorso
array[1..num_clients+1, 1..num_clients+1, 1..num_vehicles] of var 0..1: x;

array [1..num_vehicles, 1..num_clients] of var 0..1: C;


% Constraint I: somma colonne = 1

constraint (
  forall(j in 1..num_clients)(
    sum(i in 1..num_vehicles)(C[i,j]) = 1
  )
);


% Constraint II: almeno un pacco per corriere

constraint (
  forall (i in 1..num_vehicles) (
    sum(j in 1..num_clients)(C[i,j]) > 0
  )
);

% Constraint di prova: dividere in maniera circa equa i pacchi
constraint (
  forall(i in 1..num_vehicles) (
    sum(j in 1..num_clients) (C[i,j]) <= num_clients / num_vehicles + 2
  )
);


% Constraint III: constraint di peso

constraint (
  forall (i in 1..num_vehicles) (
    sum(j in 1..num_clients)(C[i,j] * size[j]) <= capacity[i]
  )
);


% Constraint IV:  channelling constraint

constraint (
  forall(j in 1..num_clients, k in 1..num_vehicles)(
    sum(i in 1..num_clients) (x[i,j,k]) == C[k,j]
  )
);

% Constraint VII: destinazione raggiunta da una partenza in cui ero
constraint(
  forall(i in 1..num_clients+1, k in 1..num_vehicles)(
    sum(j in 1..num_clients+1)(x[i,j,k]) == sum(j in 1..num_clients+1)(x[j,i,k])
  )
);

% Constraint V: constraint dei depositi (partenze)

constraint (
  forall(k in 1..num_vehicles)(
    sum(j in 1..num_clients+1)(x[num_clients+1, j, k]) == 1
  )
);

% Constraint VI: constraint dei depositi (arrivi)

constraint (
  forall(k in 1..num_vehicles)(
    sum(i in 1..num_clients+1)(x[i, num_clients+1, k]) == 1
  )
);


% Constraint VIII: diagonale a zero

constraint (
  forall(k in 1..num_vehicles)(
    sum(i in 1..num_clients+1) (x[i,i,k]) == 0
  )
);


% Calcola il percorso totale per ogni veicolo e assegna il risultato a theta[k]
constraint (
  forall (k in 1..num_vehicles) (
    theta[k] = sum(i in 1..num_clients+1, j in 1..num_clients+1)(distances[i,j] * x[i,j,k])
  )
);

% Vincola max_theta ad essere il massimo tra i valori di theta
constraint (
  forall (k in 1..num_vehicles) (
    max_theta >= theta[k]
  )
);


solve minimize max_theta;