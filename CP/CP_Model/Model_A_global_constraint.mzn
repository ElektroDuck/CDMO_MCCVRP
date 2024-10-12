%import

include "globals.mzn";

% Modelling of the instance

int: num_vehicles; %number of vehicles
int: num_clients; %number of clients
set of int: clients = 1..num_clients; %nodes rapresenting clients
set of int: vehicles = 1..num_vehicles; %set of the vehicles
array[vehicles] of int: capacity; %capacity of every vehicle
array[clients] of int: size; %size of every item

set of int: nodes = 1..num_clients+1; %nodes of the graph
array[nodes, nodes] of int: distances; %matrix of distances between nodes

% Modelling of the Decision Variables
array[vehicles, nodes] of var nodes: successor; % Building a matrix to define the paths


array[clients] of var 0..num_vehicles: loads;
%array[vehicles] of var d_low_bound..up_bound: obj_dist;
array[vehicles] of var int: obj_dist;

int : max_load = max(capacity) + 1;
array[vehicles] of var 0..max_load: couriers_loads;

var int: maximum; 

% Constraints

% ensure that each courier starts and ends in origin point
constraint forall(k in vehicles) (
      if successor[k, num_clients+1] != num_clients+1 then count([successor[k, i] | i in clients], num_clients+1, 1) 
      else forall(i in nodes) (successor[k, i] == i)  endif
);

% each item must be assigned only to one courier
constraint forall(i in clients)(
      count([successor[k, i]| k in vehicles], i, num_vehicles-1)
);

% setting the loads to verify the load_size for each courier
constraint forall(k in vehicles) (
          forall(i in clients)(
          if successor[k, i] != i then
          loads[i] = k endif
          )
);

% ensure that each courier does not exceed the max capacity
constraint bin_packing_capa(
                           capacity,
                           loads,
                           size);

% Circuit
constraint forall(k in vehicles) (
      subcircuit([successor[k, i] |i in nodes]) 
);


% distance computation
constraint forall(k in vehicles) (
      obj_dist[k] = sum([distances[i, successor[k, i]]| i in nodes where successor[k, i] != i])
);


% Symmetry breaking constraint (da sistemare)
%constraint forall(k in vehicles)(
%    couriers_loads[k] = sum(
%            [item_size[i] | i in ITEMS where loads[i]=k]
%    )
%);


% sub tour elimination, if the last courier can carry all the packages
% then all courier must start
constraint forall(k in vehicles) (
                  if capacity[num_vehicles] >= max(size) then 
                  successor[k, num_clients+1] != num_clients+1   
                  endif
            );


constraint maximum = max(obj_dist);


%var int: max_dist = max(j in vehicles)(sum(i in nodes)(distances[i, successor[j,i]]));
solve minimize maximum;

output [
    "Successor Matrix:\n \t nodes: ", 
    show([i | i in nodes]),
    "\n \t\t",
    concat([show([successor[k, j] | j in nodes]) ++ "\n\t\t" | k in vehicles]),
    
    "\n loads = \t", show(loads), "\n",
    
    "Maximum Distance = ", show(maximum), "\n"
];
