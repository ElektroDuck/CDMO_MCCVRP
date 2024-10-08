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

%All the vehicle must come back to the depo
num_vehicles = sum(last); 


array[int] of int: freq_pred = [if i == num_clients+1 then num_vehicles else is_pred[i] endif | i in nodes]; % lower bound array
constraint global_cardinality(pred, [i | i in vehicles], freq_pred);




%possibile redundant constraint
%The depo must be pred of exactly num_vehicles nodes -> every vehicle starts the route from the depo 
%Each node must appear at most 1 time in the pred array -> assure that every node has a different pred -> no same pred for 2 different nodes
array[int] of int: lb = [if i == num_clients+1 then num_vehicles else 0 endif | i in nodes]; % lower bound array
array[int] of int: ub = [if i == num_clients+1 then num_vehicles else 1 endif | i in nodes]; % upper bound array
constraint global_cardinality_low_up(pred, [i | i in vehicles], lb, ub);




%Objective function
var int: max_dist = max(j in vehicles)(sum(i in clients where vehicle[i] = j)(distances[pred[i], i] +  (distances[i, num_clients + 1]*last[i] )));

solve minimize max_dist;

