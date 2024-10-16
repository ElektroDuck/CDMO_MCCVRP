int: num_vehicles;
int: num_clients;

array[1..num_vehicles] of int: capacity;
array[1..num_clients] of int: size;
array[1..num_clients+1, 1..num_clients+1] of int: distances;                         
array[1..num_vehicles] of var int: theta;  % Ogni veicolo ha un proprio theta
var int: max_theta;  % Variabile per il massimo percorso
array[1..num_vehicles, 1..num_clients+1] of var 0..num_clients+1: x;
array [1..num_vehicles, 1..num_clients] of var 0..1: C;
var int: M = num_clients+2;



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




% Constraint III: constraint di peso

constraint (
  forall (i in 1..num_vehicles) (
    sum(j in 1..num_clients)(C[i,j] * size[j]) <= capacity[i]
  )
);

% Channelling constraints
constraint (
  forall(i in 1..num_vehicles, j in 1..num_clients)(
    x[i,j] <= C[i,j] * M
  )
);

constraint (
  forall(i in 1..num_vehicles, j in 1..num_clients)(
    x[i,j] >= C[i,j]
  )
);


% diagonale != 0
constraint (
  forall (i in 1..num_vehicles, j in 1..num_clients+1) (
    x[i,j] != j
  )
);

% Parto da un deposito
constraint (
  forall(i in 1..num_vehicles) (
    x[i, num_clients+1] > 0
  )
);

% arrivo in un deposito
constraint (
  forall(i in 1..num_vehicles) (
    sum(j in 1..num_clients+1) (x[i,j] > num_clients) > 0 
  )
);

% aiuto

constraint (
  forall(i in 1..num_vehicles, j in 1..num_clients+1 where x[i,j] != 0 /\ x[i,j] < num_clients) (
    x[i,x[i,j]] != j 
  )
);

constraint (
  forall(i in 1..num_vehicles, j in 1..num_clients+1 where x[i,j] != 0 /\ x[i,j] < num_clients) (
    x[i,x[i,j]] != 0
  )
);

constraint (
  forall( i in 1..num_vehicles, j in 1..num_clients+1, k in j..num_clients+1 where j!=k /\ x[i,j] != 0) (
    x[i,j] != x[i,k]
  )
);


% Funzione obiettivo
constraint (
  forall (i in 1..num_vehicles) (
    theta[i] = sum(j in 1..num_clients+1 where x[i,j] != 0)(distances[j,x[i,j]])
  )
);

% Vincola max_theta ad essere il massimo tra i valori di theta
constraint (
  forall (k in 1..num_vehicles) (
    max_theta >= theta[k]
  )
);


solve minimize max_theta;