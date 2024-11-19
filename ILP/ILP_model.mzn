int: num_vehicles;
int: num_clients;

array[1..num_vehicles] of int: capacity;
array[1..num_clients] of int: size;
array[1..num_clients+1, 1..num_clients+1] of int: distances;                         
array[1..num_clients+1, 1..num_clients+1, 1..num_vehicles] of var 0..1: x;
array[1..num_clients+1, 1..num_vehicles] of var 0..1: y;
array[1..num_clients, 1..num_vehicles] of var 1..num_clients: u;

var int: max_distance;

% Bound the min and max distance


% Each item is assigned to a courier
constraint (
  forall (i in 1..num_clients) (
    sum(k in 1..num_vehicles)(y[i,k]) == 1
  )
);


% Each courier starts from a depot

constraint (
  sum(k in 1..num_vehicles)(y[num_clients+1, k]) = num_vehicles
);

% Capacity constraint

constraint (
  forall(k in 1..num_vehicles) (
    sum(i in 1..num_clients) (y[i,k] * size[i]) <= capacity[k]
  )
);

% Unique path for each courier

constraint (
  forall (i in 1..num_clients, k in 1..num_vehicles) (
    sum(j in 1..num_clients+1)(x[i,j,k]) == sum(j in 1..num_clients+1)(x[j,i,k])
  )
);

% Channelling constraint

constraint (
  forall (i in 1..num_clients, k in 1..num_vehicles) (
    sum(j in 1..num_clients+1)(x[j,i,k]) == y[i,k]
  )
 );


% Diagonal = 0 (no loop)

constraint (
  forall (k in 1..num_vehicles) (
    sum(j in 1..num_clients+1) (x[j,j,k]) == 0
  )
);

% Each courier must start from a node that it has reached

constraint (
  forall (j in 1..num_clients) (
    sum(i in 1..num_clients+1, k in 1..num_vehicles) (x[i,j,k]) == 1
  )
);

% Each courier goes back into the depot

constraint (
  forall (k in 1..num_vehicles) (
    sum(i in 1..num_clients+1) (x[i,num_clients+1,k]) == 1
  )
);

% For every client, there must be a courier that reach it

constraint (
  forall (j in 1..num_clients) (
    sum (i in 1..num_clients+1, k in 1..num_vehicles) (x[i,j,k]) == 1
  )
);

% For every client, there must be a courier that reach it

constraint (
  forall (i in 1..num_clients) (
    sum (j in 1..num_clients+1, k in 1..num_vehicles) (x[i,j,k]) == 1
  )
);

% the number of courier that enters a node must be equal to the number of courier thate exit the node

constraint (
  forall (k in 1..num_vehicles, j in 1..num_clients+1)(
    sum(i in 1..num_clients+1) (x[i,j,k]) == sum(i in 1..num_clients+1) (x[j,i,k])
  )
);

% MTZ formulation
constraint (
  forall (i in 1..num_clients, j in 1..num_clients, k in 1..num_vehicles where i != j) (
    u[i,k] - u[j,k] + num_clients * x[i,j,k] <= num_clients - 1
  )
);

% Objective function
constraint (
  forall (k in 1..num_vehicles) (
    sum(i in 1..num_clients+1, j in 1..num_clients+1) (x[i, j, k] * distances[i, j]) <= max_distance
  )
);



solve minimize max_distance;

output [show(max_distance), show("|"), show(x)]; 