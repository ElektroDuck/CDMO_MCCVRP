include "globals.mzn";

int: num_clients; % Item number
int: num_vehicles; % Vehicle number
array [1..num_vehicles] of int: capacity; % Vehicle capacities
array [1..num_clients] of int: size; % Item sizes (demand)

set of int: NODES = 1..num_clients+1;
set of int: CUSTOMER = 1..num_clients;
set of int: VEHICLE = 1..num_vehicles;

array [NODES,NODES] of int: distances;

array [CUSTOMER] of var NODES: pred;
array [CUSTOMER] of var 0..1: last;
array [CUSTOMER] of var 0..1: is_pred;
array [CUSTOMER] of var VEHICLE: vehicle;
array [CUSTOMER] of var CUSTOMER: num_visit;
var VEHICLE: used_couriers;

%1. Number of used vehicles should be equal to the number of vehicles returning to the depot
constraint used_couriers = sum(i in CUSTOMER)(last[i]);

%2. Check that: 
%     - if a customer i has is_pred set to 1 then it is contained exactly once in the pred array
%     - the number of vehicles leaving the depo is equal to the occurency of the depo idx in the pred array
constraint global_cardinality(pred, NODES, is_pred ++ [used_couriers]);

%
constraint forall(i in CUSTOMER)(if pred[i] != num_clients+1 then vehicle[i] == vehicle[pred[i]] endif);

%Requires that the number of distinct values in vehicle should be the number of used couriers
constraint nvalue(used_couriers, vehicle);

%max number in the vehicle array should be equal to the number of used vehicles
constraint max(vehicle) = used_couriers;

%a node cannot be both a predecessor and a last node
constraint forall(i in CUSTOMER)(is_pred[i] + last[i] = 1);

%a customer cannot be its own predecessor
constraint redundant_constraint(forall(i in CUSTOMER)(pred[i] != i));

%we need a subtour constraint(single,circular tour)
constraint forall(i in CUSTOMER)(if pred[i] != num_clients+1 then num_visit[i] > num_visit[pred[i]] endif);

%capacity constraint
constraint forall(j in 1..used_couriers)((sum(i in CUSTOMER where vehicle[i]==j)(size[i])) <= capacity[j]);

%Objective function
var int: max_dist = max(j in 1..used_couriers)(sum(i in CUSTOMER where vehicle[i]==j)(distances[pred[i],i] + distances[i,num_clients+1]*last[i]));


solve minimize max_dist;
 
output [
    "predecessor = \t", show(pred), "\n",
    "num_visit = \t", show(num_visit), "\n",
    "last = \t\t", show(last), "\n",
    "is_pred = \t", show(is_pred), "\n",
    "vehicle = \t", show(vehicle), "\n",
    "total_couriers = \t", show(num_vehicles), "\n",
    "used_couriers = \t", show(used_couriers), "\n",
    "Maximum Distance = \t", show(max_dist), "\n"
];


