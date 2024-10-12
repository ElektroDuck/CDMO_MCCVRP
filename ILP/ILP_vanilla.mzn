int: num_vehicles;
int: num_clients;

array[1..num_vehicles] of int: capacity;
array[1..num_clients] of int: size;
array[1..num_clients+1, 1..num_clients+1] of int: distances;

array[1..num_vehicles] of var int: theta;  % Ogni veicolo ha un proprio theta
var int: max_theta;  % Variabile per il massimo percorso
array[1..num_clients+1, 1..num_clients+1, 1..num_vehicles] of var 0..1: x;


% La destinazione è raggiunta solamente da un corriere
constraint (
  forall (j in 1..num_clients) (
    sum(k in 1..num_vehicles, i in 1..num_clients)(x[i,j,k]) == 1
  )
);

% Forzare la diagonale a zero, per evitare loop nei punti di delivery
constraint (
  forall (k in 1..num_vehicles) (
    sum(i in 1..num_clients+1)(x[i,i,k]) == 0
  )
);

% Un corriere parte ed arriva al deposito
constraint (
  forall (k in 1..num_vehicles) (
    sum(j in 1..num_clients+1)(x[num_clients+1,j,k]) == 1
  )
);

constraint (
  forall (k in 1..num_vehicles) (
    sum(i in 1..num_clients+1)(x[i,num_clients+1,k]) == 1
  )
);

% Constraint di peso
constraint (
  forall (k in 1..num_vehicles) (
    sum(i in 1..num_clients, j in 1..num_clients)(x[i,j,k] * size[j]) <= capacity[k]
  )
);

% Controlla che un corriere parta da una destinazione nel quale è stato
% "Un corriere segue un percorso senza teletrasportarsi"
constraint (
  forall (k in 1..num_vehicles, i in 1..num_clients+1) (
    sum(j in 1..num_clients+1)(x[i,j,k]) == sum(j in 1..num_clients+1)(x[j,i,k])
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

% Funzione obiettivo: minimizzare il percorso massimo
solve minimize max_theta;
