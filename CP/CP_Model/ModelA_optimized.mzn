%import

include "globals.mzn";

% Modelling of the instance

int: num_vehicles; %number of vehicles
int: num_clients; %number of clients
set of int: nodes = 1..num_clients+1; %nodes of the graph
set of int: clients = 1..num_clients; %nodes rapresenting clients
set of int: vehicles = 1..num_vehicles; %set of the vehicles
array[vehicles] of int: capacity; %capacity of every vehicle
array[clients] of int: size; %size of every item
array[nodes, nodes] of int: distances; %matrix of distances between nodes

% Modelling of the Decision Variables

array[vehicles, nodes] of var nodes: successor; % Building a matrix to define the paths

% Constraints

