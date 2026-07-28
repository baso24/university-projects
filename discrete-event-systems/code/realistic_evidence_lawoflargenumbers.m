%Script for giving evidence of Law of Large Numbers, Real clock structure case [Basili Valentino]
clear all
close all
clc

% Parameters
% Real case
Td = 1.5; % [h]
lambda_a = 1;  % [train arrivals/h]
lambda_g = 0.5; % [faults/h]
min_c = 0.2;
max_c = 0.4;
min_r = 0.5;
max_r = 1.5; 
pi0 = [ 1 0 0 0 0 0 0 0 0 0 0 ]; % initial state
t_star = 30; % time of interest

% Definition of parameters
m = 5; % m is the number of events
n = 11; % n is the number of states
ename = {'a','d','c','g','r'}; % original names of the events
xname = {'000','001','002','102','012','011','101','100','110','111','112'}; % original names of the states

% Definition of the logical model
model.m = m;
model.n = n;
% transition probabilities
model.p = zeros(n, n, m);
% Inizializzazione con self-loop per garantire PMF valide su tutte le colonne
for x = 1:n
    for e = 1:m
        model.p(:, x, e) = NaN(n, 1); 
    end
end

% EVENTO 1: 'a'
model.p(:, 1, 1) = [ 0 ; 1 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ]; 
model.p(:, 2, 1) = [ 0 ; 0 ; 0 ; 0 ; 0 ; 1 ; 0 ; 0 ; 0 ; 0 ; 0 ]; 
model.p(:, 3, 1) = [ 0 ; 0 ; 0 ; 0 ; 1 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ]; 
model.p(:, 4, 1) = [ 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 1 ]; 
model.p(:, 7, 1) = [ 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 1 ; 0 ]; 
model.p(:, 8, 1) = [ 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 1 ; 0 ; 0 ]; 
% EVENTO 2: 'd'
model.p(:, 3, 2) = [ 1 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ]; 
model.p(:, 5, 2) = [ 0 ; 1 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ]; 
% EVENTO 3: 'c'
model.p(:, 2, 3) = [ 0 ; 0 ; 1 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ]; 
model.p(:, 6, 3) = [ 0 ; 0 ; 0 ; 0 ; 1 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ]; 
% EVENTO 4: 'g'
model.p(:, 1, 4) = [ 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 1 ; 0 ; 0 ; 0 ];
model.p(:, 2, 4) = [ 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 1 ; 0 ; 0 ; 0 ; 0 ]; 
model.p(:, 3, 4) = [ 0 ; 0 ; 0 ; 1 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ]; 
model.p(:, 5, 4) = [ 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 1 ]; 
model.p(:, 6, 4) = [ 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 1 ; 0 ]; 
% EVENTO 5: 'r'
model.p(:, 4, 5) = [ 0 ; 0 ; 1 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ]; 
model.p(:, 7, 5) = [ 0 ; 1 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ]; 
model.p(:, 8, 5) = [ 1 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ]; 
model.p(:, 9, 5) = [ 0 ; 1 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ]; 
model.p(:, 10,5) = [ 0 ; 0 ; 0 ; 0 ; 0 ; 1 ; 0 ; 0 ; 0 ; 0 ; 0 ]; 
model.p(:, 11,5) = [ 0 ; 0 ; 0 ; 0 ; 1 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ]; 

% Initial state
model.p0 = pi0'; % initial state probabilities (vector 11 x 1)

% Definition of the stochastic clock structure
% Real case
F{1} = 'exprnd(1/lambda_a, 1, L)';
F{2} = 'Td*ones(1,L)'; % L values equal to Td
F{3} = 'unifrnd(min_c, max_c, 1, L)'; % L values drawn from U(A,B)
F{4} = 'exprnd(1/lambda_g, 1, L)'; % L values drawn from U(C,D)
F{5} = 'unifrnd(min_r, max_r, 1, L)'; % L values equal to Tr

% MULTIPLE SIMULATIONS
disp('MULTIPLE SIMULATIONS'), disp(' ')

M = 100; % Number of replications 
% each k is the number of events until we stop the simulation
% so we will do:
% 100 simulations that stops after 10 events
% 100 simulations that stops after 50 events
% 100 simulations that stops after 100 events
% ... and so on
kmax_values = [10, 50, 100, 250, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000];

% means and variances matrices
means = zeros(length(kmax_values), n);
variances = zeros(length(kmax_values), n);

for idx = 1:length(kmax_values)
    k = kmax_values(idx);
    
    % temporary matrix for all of the states' estimates of this M
    states_estimate = zeros(M, n);
    
    % M independent simulations, k events long
    for j = 1:M
        % Definition of the clock sequences
        L = k; % length of the clock sequences
        V = [];
        for e = 1:m
            eval([ 'V(' num2str(e) ',:) = ' F{e} ';' ]);
        end
        
        % Simulation
        [E, X, T] = simprobdes(model, V);
        
        % diff(T) calcola i tempi di permanenza di ogni stato
        % così una volta che so QUANDO sono stato in uno stato x, posso
        % cercare dentro questo vettore per sapere QUANTO ci sono stato
        tempi_di_permanenza = diff(T);
        
        % calcolo le stime per tutti gli 11 stati
        for x = 1:n
            % troviamo gli intervalli in cui il sistema era nello stato x
            indici_stato = (X(1:end-1) == x);
            
            % sommo i tempi spesi nello stato x e divido per il tempo
            % totale della simulazione T(end)
            tempo_totale_in_stato_x = sum(tempi_di_permanenza(indici_stato));
            states_estimate(j, x) = tempo_totale_in_stato_x / T(end);
        end
    end
    
    % dopo le M simulazioni, rispetto a questo k (numero di eventi)
    % calcolo media e varianza
    % ogni riga contiene la media o varianza degli stati in tutte le M simulazioni
    % alla fine means e variances avranno dimensione di length(kmax_values) righe e 11 colonne
    means(idx, :) = mean(states_estimate);
    variances(idx, :) = var(states_estimate);
    
    disp(['Completed kmax = ' num2str(k)]);
end

figure('Name', 'Evidence of Law of Large Numbers');

% subplot variance
subplot(2,1,1);
plot(kmax_values, variances, '-');
grid on;
title('Sample variance');
xlabel('Number of simulated events');
ylabel('Variance of the estimates');
legend(xname, 'Location', 'eastoutside'); % legenda fuori dal grafico

% subplot mean
subplot(2,1,2);
plot(kmax_values, means, '-');
grid on;
title('Sample mean');
xlabel('Number of simulated events');
ylabel('Mean of the estimates');
legend(xname, 'Location', 'eastoutside'); % legenda fuori dal grafico