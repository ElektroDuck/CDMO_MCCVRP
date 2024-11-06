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
int: low_bound; %lower bound
int: up_bound; %upper bound
int: min_dist_bound; %minumum distance every vehicle has to travel

% Modelling of the Decision Variables
array[vehicles, nodes] of var nodes: successor; % Building a matrix to define the paths
array[clients] of var 0..num_vehicles: loads;
%array[vehicles] of var d_low_bound..up_bound: obj_dist;
array[vehicles] of var min_dist_bound..up_bound: vehicle_dist;
int : max_load = max(capacity);
array[vehicles] of var 0..max_load: couriers_loads;
var low_bound..up_bound: maximum; 
% Constraints

%- 1 Circuit, every path must be closed
constraint forall(vehicle in vehicles) (
      subcircuit(successor[vehicle, nodes]));
      
constraint forall(vehicle in vehicles) (
      alldifferent(successor[vehicle,nodes]));

%2. Ensure that each courier starts and ends in origin point

%- 2.1 Ensure that all the vehicle leave the depo by forcing the last column of the matrix to be different from the depo idx
constraint forall(veicle in vehicles)(successor[veicle, num_clients+1] != num_clients+1); 

%- 2.2 Ensure that the vehicle come back to the depo
constraint forall(vehicle in vehicles)(count(successor[vehicle, clients], num_clients+1, 1));


%- 3 Ensure each item is assigned only to one courier
constraint forall(client in clients)(
      count(successor[vehicles, client], client, num_vehicles-1)
);

%- 4 Checking the loads

%- 4.1 Setting the loads to verify the load_size for each courier
constraint forall(vehicle in vehicles) (
          forall(client in clients)(
          if successor[vehicle, client] != client then
          loads[client] = vehicle endif
          )
);
%- 4.2 ensure that each courier does not exceed the max capacity
constraint bin_packing_capa(
                           capacity,
                           loads,
                           size);

%SIMMETRY BREAKING

% Symmetry breaking based on loads for couriers with the same capacity
constraint forall(v in 1..num_vehicles, ov in v..num_vehicles where capacity[v]==capacity[ov])
    ( couriers_loads[v] >= couriers_loads[ov]);
    
%OBJECTIVE FUNCTION                           

% distance computation
constraint forall(vehicle in vehicles) (
      vehicle_dist[vehicle] = sum([distances[i, successor[vehicle, i]]| i in nodes]) %where successor[vehicle, i] != i])
);
% Channeling constraint, define the loads of the couriers
constraint forall(vehicle in vehicles)(
    couriers_loads[vehicle] = sum(
            [size[client] | client in clients where loads[client]=vehicle]
    )
);
% sub tour elimination, if the last courier can carry all the packages
% then all courier must start
%constraint forall(k in vehicles) (
%                  if capacity[num_vehicles] >= max(size) then 
%                  successor[k, num_clients+1] != num_clients+1   
%                  endif
%            );

constraint maximum = max(vehicle_dist);
%var int: max_dist = max(j in vehicles)(sum(i in nodes)(distances[i, successor[j,i]]));
solve 
:: int_search(successor, dom_w_deg, indomain_random) %for chuffed solver comment this line (not implemented by minizinc yet)
minimize maximum;


%output [
%    "Successor Matrix:\n \t nodes: ", 
%    show([i | i in nodes]),
%    "\n \t\t",
%    concat([show([successor[k, j] | j in nodes]) ++ "\n\t\t" | k in vehicles]),
%    
%    "\n loads = \t", show(loads), "\n",
%    
%    "Maximum Distance = ", show(maximum), "\n"
%];

output [show(successor), "|", show(maximum)]; 