%MODEL 2

%import
include "globals.mzn";

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
array[clients] of var nodes: pred; %what node is the predecessor of the i-th node
array[clients] of var bool: last; %if the node is the last of a route
array[clients] of var bool: is_pred; %if the node is the predecessor of another (if it is not last)
array[clients] of var vehicles: vehicle; %which vehicle is the one assigned to the i-th item

array[clients] of var 1..max(capacity): load;

%Constraints
constraint sum(last)=num_vehicles; %we use all the vehicles
% so the number of last packages must be equal to the number of vehicles

constraint global_cardinality(pred, nodes, is_pred ++ [num_vehicles]); %we check that every clients is at most one time in the pred array
% and that the depot appears a number of times equal to the number of vehicles used 

constraint forall(i in clients) (bool_not(is_pred[i], last[i])); %if a client is a predecessor
% it can't be the last

constraint forall(i in clients) (if pred[i]!=num_clients+1 then vehicle[i]=vehicle[pred[1]] endif); %check that the items on the same route are on the same vehicle

constraint forall(i in clients) (if pred[i]==6 then load[i]+size[i]<=capacity[vehicle[i]]endif);%the load must be always less than the capacity
%of the vehicle assigned

%constraint forall(i in clients) (load[i]=load[pred[i]]-size[i]); %This constraint checks that there are no subtours


%Objective function
var int: max_dist = max(j in vehicles)(sum( i in clients where vehicle[i]=j)(distances[pred[i],i]+distances[i,num_clients+1]*last[i]));

solve minimize max_dist;

