% =========================================================================
% Segway Controller Simulation - Part (d)
% Desired linear velocity: vd = 1 m/sec[cite: 3]
% =========================================================================

clear; clc; close all;

% Controller gains calculated in Part (b)
k1 = 17;
k2 = 22;
k3 = -16;

% Closed-loop state-space matrices: dot{x} = A_cl*x + B_cl*vd
% State vector: x = [theta; dot(theta); v]
A_cl = [  0,    1,    0;
    1-k1, -k2,  -k3;
    -k1, -k2,  -k3 ];

B_cl = [  0;
    k3;
    k3 ];

% Outputs to plot: [theta(t); v(t)]
C_cl = [ 1, 0, 0;   % Angular displacement theta
    0, 0, 1 ]; % Linear velocity v

D_cl = [ 0; 
    0 ];

% Create state-space system
sys_cl = ss(A_cl, B_cl, C_cl, D_cl);

% Simulation time vector (0 to 5 seconds)
t = linspace(0, 5, 1000);

% Step input for desired velocity vd = 1 m/sec
vd = 1;
[y, t_out] = step(vd * sys_cl, t);

% =========================================================================
% Plotting Results
% =========================================================================
figure('Name', 'Segway Controller Simulation', 'Color', 'w');

% Plot Angular Displacement theta(t)
subplot(2, 1, 1);
plot(t_out, y(:, 1), 'b-', 'LineWidth', 1.8);
grid on;
xlabel('Time $t$ (s)', 'Interpreter', 'latex');
ylabel('$\theta(t)$ (rad)', 'Interpreter', 'latex');
title('Segway Response to Step Change in Desired Velocity ($v_d = 1$ m/s)', 'Interpreter', 'latex');
set(gca, 'FontSize', 11);

% Plot Linear Velocity v(t)
subplot(2, 1, 2);
plot(t_out, y(:, 2), 'r-', 'LineWidth', 1.8);
grid on;
xlabel('Time $t$ (s)', 'Interpreter', 'latex');
ylabel('$v(t)$ (m/s)', 'Interpreter', 'latex');
set(gca, 'FontSize', 11);