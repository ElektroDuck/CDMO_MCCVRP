int: m;
int: n;

array[1..m] of int: trucks;
array[1..n] of int: S;
array[1..n+1, 1..n+1] of int: D;                         
var int: theta;
array[1..n+1, 1..n+1, 1..m] of var 0..1: x;


% L'idea è quella di avere un vettore n per ogni corriere m, poi si riempiono seguendo le regole di peso
% | 0, 0, 0, 0, 0 |
% | 0, 0, 0, 0, 0 |
% | 0, 0, 0, 0, 0 | -> viene fuori una roba di questo tipo, dove ogni riga è un corriere
array[1..m, 1..n] of var 0..1: assignment;

% In totale nella matrice, devono esserci un numero di 1 pari al numero di pacchi
constraint(
  sum(i in 1..m, j in 1..n)(assignment[i,j]) = n
);

% Non ci devono essere pacchi assegnati a corrieri diversi (faccio una somma per colonne)
constraint(
  forall (j in 1..n)(
    sum (i in 1..m)(assignment[i,j]) = 1
  )
);

% Vincoli dei pesi (per ogni corriere, sommo gli indici a uno moltiplicati per il peso del pacco)
constraint (
  forall (i in 1..m)(
    sum (j in 1..n) (assignment[i,j] * S[j]) <= trucks[i]
  )
);


% Vincolo che collega knapsack alla matrice ma non funziona

constraint (
  forall(j in 1..n-1, k in 1..m) (
    sum(i in 1..n-1) (x[i,j,k]) == assignment[k,j]
  )
);


% Controlla che la somma dello slice di ogni colonna (per tutta la matrice) sia uguale a 1
% "Ogni destinazione è raggiunta da un solo corriere"
% constraint (
%   forall (j in 1..n) (
%     sum(k in 1..m, i in 1..n+1)(x[i,j,k]) == 1
%   )
% );


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



% Funzione obiettivo
constraint (
    forall (k in 1..m) (
      sum(i in 1..n+1, j in 1..n+1)(D[i,j] * x[i,j,k]) = theta
  )
);


solve minimize theta;