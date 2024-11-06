int: num_vehicles;
int: num_clients;

array[1..num_vehicles] of int: capacity;
array[1..num_clients] of int: size;
array[1..num_clients+1, 1..num_clients+1] of int: distances;                         

array[1..num_clients+1, 1..num_clients+1, 1..num_vehicles] of var 0..1: x;
array[1..num_clients+1, 1..num_vehicles] of var 0..1: y;
array[1..num_clients, 1..num_vehicles] of var 1..num_clients: u;


var int: max_distance;



% Each item solo ad un corriere
constraint (
  forall (i in 1..num_clients) (
    sum(k in 1..num_vehicles)(y[i,k]) == 1
  )
);


% Ogni corriere è attivo (parte da un deposito) (non l'ho capito)

constraint (
  sum(k in 1..num_vehicles)(y[num_clients+1, k]) = num_vehicles
);

% Ogni corriere rispetta capacità massima

constraint (
  forall(k in 1..num_vehicles) (
    sum(i in 1..num_clients) (y[i,k] * size[i]) <= capacity[k]
  )
);

% unicità dei travel dei corrieri
constraint (
  forall (i in 1..num_clients, k in 1..num_vehicles) (
    sum(j in 1..num_clients+1)(x[i,j,k]) == sum(j in 1..num_clients+1)(x[j,i,k])
  )
);

% Channelling tra x ed y
constraint (
  forall (i in 1..num_clients, k in 1..num_vehicles) (
    sum(j in 1..num_clients+1)(x[j,i,k]) == y[i,k]
  )
 );


% Diagonal = 0
constraint (
  forall (k in 1..num_vehicles) (
    sum(j in 1..num_clients+1) (x[j,j,k]) == 0
  )
);

% Partendo da un nodo, ogni altro corriere deve andare in un altro (???)
constraint (
  forall (j in 1..num_clients) (
    sum(i in 1..num_clients+1, k in 1..num_vehicles) (x[i,j,k]) == 1
  )
);

% Ogni corriere torna al depot
constraint (
  forall (k in 1..num_vehicles) (
    sum(i in 1..num_clients+1) (x[i,num_clients+1,k]) == 1
  )
);

% Per ogni cliente j, ci deve essere solo un veicolo k che va da j ad i
constraint (
  forall (j in 1..num_clients) (
    sum (i in 1..num_clients+1, k in 1..num_vehicles) (x[i,j,k]) == 1
  )
);

% Per ogni cliente j, ci deve essere slo un veicolo k che da i va a j

constraint (
  forall (i in 1..num_clients) (
    sum (j in 1..num_clients+1, k in 1..num_vehicles) (x[i,j,k]) == 1
  )
);

% Se un veicolo k visita j, lo stesso veicolo dovrebbe andare da j in un'altra location.
% Il numero di corrieri che entra in un nodo deve essere uguale al numero che esce

constraint (
  forall (k in 1..num_vehicles, j in 1..num_clients+1)(
    sum(i in 1..num_clients+1) (x[i,j,k]) == sum(i in 1..num_clients+1) (x[j,i,k])
  )
);

%MTZ formulation
constraint (
  forall (i in 1..num_clients, j in 1..num_clients, k in 1..num_vehicles where i != j) (
    u[i,k] - u[j,k] + num_clients * x[i,j,k] <= num_clients - 1
  )
);

% Vincolo per calcolare la distanza totale per ogni veicolo e legarla a max_distance
constraint (
  forall (k in 1..num_vehicles) (
    sum(i in 1..num_clients+1, j in 1..num_clients+1) (x[i, j, k] * distances[i, j]) <= max_distance
  )
);



solve minimize max_distance;