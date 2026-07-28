% Script for estimating state probabilities [Basili Valentino]
clear all
close all
clc

% Parameters
lambda_a = 1;  % [train arrivals/h]
lambda_d = 2;  % [train departs/h]
lambda_c = 1.5;  % [train_preparation/h]
lambda_g = 0.5;  % [fault/h]
lambda_r = 0.5;  % [repairs/h]
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
% Inizializzazione 
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
F{1} = 'exprnd(1/lambda_a,1,L)'; % L values drawn from Exp(1/lambda_a)
F{2} = 'exprnd(1/lambda_d,1,L)'; % L values drawn from Exp(1/lambda_d)
F{3} = 'exprnd(1/lambda_c,1,L)'; % L values drawn from Exp(1/lambda_c)
F{4} = 'exprnd(1/lambda_g,1,L)'; % L values drawn from Exp(1/lambda_g)
F{5} = 'exprnd(1/lambda_r,1,L)'; % L values drawn from Exp(1/lambda_r)

% MULTIPLE SIMULATIONS
disp('MULTIPLE SIMULATIONS'), disp(' ')

% Parameters of the simulations
kmax = 250; % maximum event index
N = 1e4; % number of simulations

% Simulations
EE = zeros(N,kmax);
XX = zeros(N,kmax+1);
TT = zeros(N,kmax+1);
disp(' Simulations in progress...')
for i = 1:N,
    % Progress
    if ismember(i,0:round(N/200):N),
        disp([ '   Progress ' num2str(i/N*100) '%' ])
    end
    
    % Definition of the clock sequences
    L = kmax; % length of the clock sequences
    V = [];
    for j = 1:m,
        eval([ 'V(' num2str(j) ',:) = ' F{j} ';' ]);
    end
    
    % Simulation
    [E,X,T] = simprobdes(model,V);
    
    % Check
    if T(end) < t_star
        error('Insufficient number of events, increase ''kmax''')
    end
    
    % Store the simulation results
    EE(i,:) = E;
    XX(i,:) = X; % matrix with dim. N x kmax
    TT(i,:) = T; % matrix with dim. N x kmax
end
disp(' Simulations completed')

% Counting how many times the system is in each state at time tstar
tol = 1e-10; % tolerance for time comparisons
nx = zeros(1,n); % zero row vector 1x11
r = (1:N)'; % Nx1 vector
c = sum(TT <= t_star+tol, 2); % Nx1 vector, sum on the rows of a boolean matrix where the ones satisfy the condition
ind = (c - 1) * N + r; % Nx1 vector, pointers to go exactly in the right position of XX
for x = 1:n % for every state
    nx(1,x) = sum(XX(ind) == x); % sum of how many times i find state x along the XX(ind)th column
end

% Estimating state probabilities at time tstar
px_est = nx/N

% Plot of state probabilities vs time
Tspan = 0:0.1:30; % grid of time values
L = length(Tspan);
nx = zeros(L,n);
r = (1:N)';
for j = 1:L
    c = sum(TT <= Tspan(j)+tol,2);
    ind = (c - 1) * N + r; % linear index
    for x = 1:n
        nx(j,x) = sum(XX(ind) == x); % counting
    end
end
PI_est = nx/N; % Estimating state probabilities 

% px_est could also be directly extracted from the last row of PI_est 
% since Tspan(end) == t_star.
% px_est = PI_est(end, :);

figure
plot(Tspan,PI_est,t_star,px_est,'*')
title('Estimated state probabilities vs time')
xlabel('t [h]')
legend('\pi_1(t)','\pi_2(t)','\pi_3(t)','\pi_4(t)','\pi_5(t)','\pi_6(t)','\pi_7(t)','\pi_8(t)','\pi_9(t)','\pi_10(t)','\pi_11(t)')


