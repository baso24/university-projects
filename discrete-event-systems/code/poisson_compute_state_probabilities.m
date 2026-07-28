% Script for computing state probabilities [Basili Valentino]
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
q1_1   = -(lambda_a + lambda_g);
q2_2   = -(lambda_c + lambda_a + lambda_g);
q3_3   = -(lambda_d + lambda_g + lambda_a);
q4_4   = -(lambda_r + lambda_a);
q5_5   = -(lambda_d + lambda_g);
q6_6   = -(lambda_c + lambda_g);
q7_7   = -(lambda_r + lambda_a);
q8_8   = -(lambda_r + lambda_a);
q9_9   = -lambda_r;
q10_10 = -lambda_r;
q11_11 = -lambda_r;

% Transition rate matrix
Q = [
    q1_1,     lambda_a, 0,        0,        0,        0,        0,        lambda_g, 0,        0,        0;
    0,        q2_2,     lambda_c, 0,        0,        lambda_a, lambda_g, 0,        0,        0,        0;
    lambda_d, 0,        q3_3,     lambda_g, lambda_a, 0,        0,        0,        0,        0,        0;
    0,        0,        lambda_r, q4_4,     0,        0,        0,        0,        0,        0,        lambda_a;
    0,        lambda_d, 0,        0,        q5_5,     0,        0,        0,        0,        0,        lambda_g;
    0,        0,        0,        0,        lambda_c, q6_6,     0,        0,        0,        lambda_g, 0;
    0,        lambda_r, 0,        0,        0,        0,        q7_7,     0,        0,        lambda_a, 0;
    lambda_r, 0,        0,        0,        0,        0,        0,        q8_8,     lambda_a, 0,        0;
    0,        lambda_r, 0,        0,        0,        0,        0,        0,        q9_9,     0,        0;
    0,        0,        0,        0,        0,        lambda_r, 0,        0,        0,        q10_10,   0;
    0,        0,        0,        0,        lambda_r, 0,        0,        0,        0,        0,        q11_11
];

% Plot of state probabilities vs time
T = 0:0.01:30;
PI = [];
for t = T
    PI = [ PI ; pi0*expm(Q*t) ];
end
figure
plot(T,PI)
title('Computed state probabilities vs time')
xlabel('t [h]')
legend('\pi_1(t)','\pi_2(t)','\pi_3(t)','\pi_4(t)','\pi_5(t)','\pi_6(t)','\pi_7(t)','\pi_8(t)','\pi_9(t)','\pi_10(t)','\pi_11(t)')

% Limit state probabilities at t=30 h
t_star = 30;
pi_star = pi0*expm(Q*t_star);
disp("Limit state probabilities computed analytically: ");
disp(['pi = [' num2str(pi_star) ']']);

