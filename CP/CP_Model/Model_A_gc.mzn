%import

include "globals.mzn";

% Input Variables

int: num_vehicles; %number of vehicles
int: num_clients; %number of clients
set of int: clients = 1..num_clients; %nodes rapresenting clients
set of int: vehicles = 1..num_vehicles; %set of the vehicles
array[vehicles] of int: capacity; %capacity of every vehicle
array[clients] of int: size; %size of every item

set of int: nodes = 1..num_clients+1; %nodes of the graph
array[nodes, nodes] of int: distances; %matrix of distances between nodes

%Model Variabels
array[vehicles, nodes] of var nodes: successor; %Building a matrix to define the paths

array[clients] of var 0..num_vehicles: assigned_vehicle; %Define an array that will store the relationship between the clients and the assigned vehicle

%array[vehicles] of var d_low_bound..up_bound: obj_dist;
array[vehicles] of var int: vehicle_distances;

int : max_load = max(capacity) + 1;
array[vehicles] of var 0..max_load: couriers_loads;

var 100..200: max_distance; 

%-------------
%CONSTRAINTS

% Path constraints 

%1. Each row in the successor matrix must represent a subcircuit. 
%   That's mean that the path represented in the row must be closed
constraint forall(veicle in vehicles) (
      subcircuit([successor[veicle, node] |node in nodes]) 
);


%2. Ensure that each courier starts and ends in origin point

%- 2.1 Ensure that all the vehicle leave the depo by forcing the last column of the matrix to be different from the depo idx
constraint forall(veicle in vehicles)(successor[veicle, num_clients+1] != num_clients+1); 

%- 2.2 Ensure that the vehicle come back to the depo
constraint forall(veicle in vehicles)(count([successor[veicle, client] | client in clients], num_clients+1, 1));


% --Load contraints-- 

%3. Esure that an item is assigned to only one vehicle. 
%   This is done cheking the count of the element on a column that are equal to the index of the column
%   If the number is equal to num_vehicles-1 that means that only one element is different -> there is only onw vehicle that starting from that position moves to an another
constraint forall(client in clients)(count([successor[vehicle, client]| vehicle in vehicles], client, num_vehicles-1));

%4. Ensure that the load carried by each vehicle doesn't exceed the limit

% In order to apply the global contraint we need to create a support array that encode the client with the assigned vehicle
constraint forall(vehicle in vehicles) (
              forall(client in clients)(
                 if successor[vehicle, client] != client then
                    assigned_vehicle[client] = vehicle endif
              )
            ); 
% use the support array in order to use the global constraint that check that  
% each item i with weight size[i, be put into assigned_vehicle[i] such that the sum of the weights of the items in each vehicle does not exceed the capacity capacity[vehicle].      
constraint bin_packing_capa(capacity, assigned_vehicle, size);


% distance computation
constraint forall(k in vehicles) (
      vehicle_distances[k] = sum([distances[i, successor[k, i]]| i in nodes where successor[k, i] != i])
);


% Symmetry breaking constraint (da sistemare)
%constraint forall(k in vehicles)(
%    couriers_loads[k] = sum(
%            [item_size[i] | i in ITEMS where loads[i]=k]
%    )
%);



constraint max_distance = max(vehicle_distances);

solve minimize max_distance;


output [
    "Successor Matrix:\n \t nodes: ", 
    show([i | i in nodes]),
    "\n \t\t",
    concat([show([successor[k, j] | j in nodes]) ++ "\n\t\t" | k in vehicles]),
    
    "\n loads = \t", show(assigned_vehicle), "\n",
    
    "Maximum Distance = ", show(max_distance), "\n"
];
