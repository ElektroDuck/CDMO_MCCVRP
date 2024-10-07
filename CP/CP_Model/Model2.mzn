%MODEL 2

%import

include "globals.mnz";

% Modelling
int: num_vehicles; %number of vehicles
int: num_clients; %number of clients
set of int: nodes = 1..num_clients+1; %nodes of the graph
set of int: clients = 1..num_clients; %nodes rapresenting clients
set of int: vehicles = 1..num_vehicles; %set of the vehicles
array[vehicles] of int: capacity; %capacity of every vehicle
array[clients] of int: size; %size of every item
array[nodes, nodes] of int: distances; %matrix of distances between nodes

%Decision Variables
array[nodes] of nodes: pred; %what node is the predecessor of the i-th node
array[nodes] of boolean: last; %if the node is the last of a route
array[nodes] of boolean: is_pred; %if the node is the predecessor of another (if it is not last)
array[nodes] of vehicles: vehicle; %which veicle is the one assigned to the i-th item

%Constraints


%Objective function
var int: max_dist = max(j in vehicles)((i in clients where vehicle(i)=j)(distances[pred[i],i]));

solve minimize max_dist;

