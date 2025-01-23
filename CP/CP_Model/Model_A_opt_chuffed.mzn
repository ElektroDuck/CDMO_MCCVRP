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

%Subcirtuit: every path must be closed
constraint forall(vehicle in vehicles) (
      subcircuit(successor[vehicle, nodes])); %Checks that every courrier has a cyclic path
constraint forall(vehicle in vehicles) (
      alldifferent(successor[vehicle,nodes])); %Redundant constraint, already implied in subcircuit

%Load Capacity: every courrier must carry less wight than its maximum capacity
constraint forall(vehicle in vehicles)(
    couriers_loads[vehicle] = sum(
            [size[client] | client in clients where loads[client]=vehicle]
    )
);    
constraint forall(vehicle in vehicles)(
    couriers_loads[vehicle]<=capacity[vehicle]
    );
    
%Every item must be taken only once
constraint forall(client in clients)(
      count(el in successor[vehicles, client])(el!=client)==1);

%Every courrier must leave the depot
constraint forall(veicle in vehicles)(successor[veicle, num_clients+1] != num_clients+1); 

%every courrier must come back to the depot
constraint forall(vehicle in vehicles)(count(successor[vehicle, clients], num_clients+1, 1));

%- 4.2 ensure that each courier does not exceed the max capacity %PROVA AD AGGIUNGERE BIN PACKING MANUALE
constraint bin_packing_capa(
                           capacity,
                           loads,
                           size);

%- 4.1 Setting the loads to verify the load_size for each courier
constraint forall(vehicle in vehicles) (
          forall(client in clients)(
          if successor[vehicle, client] != client then
          loads[client] = vehicle endif
          )
);
    
%OBJECTIVE FUNCTION                           

% distance computation
constraint forall(vehicle in vehicles) (
      vehicle_dist[vehicle] = sum([distances[i, successor[vehicle, i]]| i in nodes])
);
%max distance
constraint maximum = max(vehicle_dist);

solve minimize maximum;

output [show(successor), "|", show(maximum)]; 