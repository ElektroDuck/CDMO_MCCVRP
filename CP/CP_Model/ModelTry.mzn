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
array[clients] of var 1..num_clients: order; % indicates the order of visit

% Constraints

constraint forall(i in clients) (count_eq(successor[..,i],i,num_vehicles-1)); %each customer visited once
constraint forall(i in vehicles)(alldifferent(successor[i,..])); %same vehicle cannot visit the same customer again
constraint forall(i in vehicles)
                  (sum(j in nodes where successor[i,j]!=j /\ successor[i,j]!=num_clients+1)(size[successor[i,j]])<= capacity[i]); %capacity constraint
                  
constraint forall(i in vehicles)
                  (forall(j in clients) (if j != successor[i,j] /\ successor[i,j] != num_clients+1 then order[j] < order[successor[i,j]] endif)); %we need a subtour constraint(single,circular tour)


                  
%implied constraint                
constraint forall(i in vehicles)
                  (if successor[i,num_clients+1]==num_clients+1 then successor[i,..]=nodes endif);
%if a vehicle does not leave depot every value of the array must be equal to the index (it does not move)

var int: max_dist = max(j in vehicles)(sum(i in nodes)(distances[i, successor[j,i]]));

solve minimize max_dist;

output [
    "Order = ", show(order), "\n",
    "Path = ", show(successor), "\n",
    "Maximum Distance = ", show(max_dist), "\n"
];