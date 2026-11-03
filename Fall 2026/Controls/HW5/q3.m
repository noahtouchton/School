clc; clear;
% System definition
A = [-2,  1;
    1, -2];
B = [1; 0];
C = eye(2);
D = [0; 0];
sys = ss(A, B, C, D);

t0 = 0;
tf = 5;
x0 = [0; 0];
xf = [1; 0];

% 1. Compute Controllability Gramian over [0, tf]
% Using Lyapunov / numeric integration
Gramian = integral(@(tau) expm(A*(tf - tau)) * (B * B') * expm(A'*(tf - tau)), ...
    0, tf, 'ArrayValued', true);

% 2. Solve for costate multiplier
eta = Gramian \ xf;

% 3. Time vector and optimal input u*(t)
N = 1000;
t = linspace(t0, tf, N)';
u = zeros(N, 1);
for k = 1:N
    u(k) = B' * expm(A' * (tf - t(k))) * eta;
end

% 4. Simulate response using lsim
[y, t_out, x] = lsim(sys, u, t, x0);

% 5. Verify terminal condition
fprintf('Target terminal state:   x(5) = [%.4f, %.4f]''\n', xf(1), xf(2));
fprintf('Achieved terminal state: x(5) = [%.4f, %.4f]''\n', x(end, 1), x(end, 2));

% 6. Plotting
figure('Color', 'w', 'Position', [150, 150, 900, 600]);

subplot(2, 1, 1);
plot(t, u, 'b-', 'LineWidth', 1.8);
grid on;
title('Minimum Norm Control Input u^*(t)');
xlabel('Time (s)');
ylabel('Control Input u(t)');

subplot(2, 1, 2);
plot(t, x(:, 1), 'r-', 'LineWidth', 1.8); hold on;
plot(t, x(:, 2), 'k--', 'LineWidth', 1.8);
yline(1, 'r:', 'LineWidth', 1.0);
yline(0, 'k:', 'LineWidth', 1.0);
grid on;
title('State Trajectories via lsim');
xlabel('Time (s)');
ylabel('States');
legend('x_1(t)', 'x_2(t)', 'Location', 'northwest');