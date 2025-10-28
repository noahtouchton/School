clc; clear; close all;

% Parameters
m = 1000; k = 2*5000; c = 7500;   % total stiffness = 10000 N/m
A = [0 1; -k/m  -c/m];            % [0 1; -10 -7.5]
x0 = [1; 0];                      % [x(0); xdot(0)]

% Poles and eigenvectors
[V,D] = eig(A);
lambda = diag(D);                 % [lambda1; lambda2] (real, negative)
v1 = V(:,1); v2 = V(:,2);

% Participation scalars from x0 = V*eta
eta = V \ x0;                     % [eta1; eta2]

% Time vector
t = linspace(0, 2, 2001).';

% Full state response with initial()
sys = ss(A, [], eye(2), []);
[y_full, t_full] = initial(sys, x0, t);

% First convergence term only, second term only, and their sum
x_first  = eta(1)*exp(lambda(1)*t);         % position from first term
x_second = eta(2)*exp(lambda(2)*t);         % position from second term
x_sum    = x_first + x_second;               % should match y_full(:,1)

% Plot per style spec
figure; hold on; grid on;
plot(t_full, y_full(:,1), 'k-', 'LineWidth', 1.5);          % black solid: initial()
plot(t, x_first,          'r:', 'LineWidth', 1.5);          % red dotted: first conv.
plot(t, x_second,         'r--', 'LineWidth', 1.5);         % red dashed: second conv.
plot(t, x_sum,            'r+', 'MarkerSize', 5);           % red symbols: sum
xlabel('Time (s)'); ylabel('Position x(t) (m)');
legend('initial() full response','first convergence','second convergence','sum of both', ...
       'Location','best');
title('Overdamped 1-DOF response decomposed into two real modes');
