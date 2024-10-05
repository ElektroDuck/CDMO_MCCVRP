int: m;
int: n;

array[1..m] of int: trucks;
array[1..n] of int: S;
array[1..n+1, 1..n+1] of int: D;


%array[1..n] of var int: aux;



                               
var int: theta;
array[1..n+1, 1..n+1, 1..m] of var 0..1: x;


% Controlla che la somma dello slice di ogni colonna (per tutta la matrice) sia uguale a 1
% "Ogni destinazione è raggiunta da un solo corriere"
constraint (
  forall (j in 1..n) (
    sum(k in 1..m, i in 1..n+1)(x[i,j,k]) == 1
  )
);


% Controlla che un corriere parta da una destinazione nel quale è stato
% "Un corriere segue un percorso senza teletrasportarsi"
constraint (
  forall (k in 1..m, i in 1..n+1) (
    sum(j in 1..n+1)(x[i,j,k]) == sum(j in 1..n+1)(x[j,i,k])
  )
);

% forzare la diagonale a zero, per evitare loop nei punti di delivery
constraint (
  forall (k in 1..m) (
    sum(i in 1..n+1)(x[i,i,k]) == 0
  )
);

% Un corriere parte ed arriva al deposito
constraint (
  forall (k in 1..m) (
    sum(j in 1..n+1)(x[n+1,j,k]) == 1
  )
);

constraint (
  forall (k in 1..m) (
    sum(i in 1..n+1)(x[i,n+1,k]) == 1
  )
);

% Constraint di peso
constraint (
  forall (k in 1..m) (
    sum(i in 1..n+1, j in 1..n)(x[i,j,k] * S[j]) <= trucks[k]
  )
);


%The approach we used to substitute the Circuit constraint used in CP is the
%following. We start by creating N auxiliary variables, u1 . . . uN , among which
%the last one has value 1, uN = 1, and the values of the remaining variables
%depend on the arcs selected. In particular, whenever an arc among the matrices
%is selected, e.g. xk,i,j , it enforces uj > ui
%.

% constraint(
%   aux[n] = 1
% );

% constraint (
%   forall(k in 1..m, i in 1..n, j in 1..n)(
%     x[i,j,k] * aux[j] >= x[i,j,k] * (1 + aux[i])
%   )
% );

% Funzione obiettivo
constraint (
    forall (k in 1..m) (
      sum(i in 1..n+1, j in 1..n+1)(D[i,j] * x[i,j,k]) = theta
  )
);


solve minimize theta;