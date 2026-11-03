clc; clear; close('all');


% f^T *x
f = [-5, -1];

% A*x <= b

A = [-1, 0;
    3, 1;
    1, -2];
b = [0; 11; 2];

% Run linprog
options = optimoptions('linprog', 'Display', 'none');
[x_star, fval, exitflag, output, lambda] = linprog(f, A, b, [], [], [], [], options);

% Display results
fprintf('--- Optimization Results (linprog) ---\n');
fprintf('Optimal x1* : %.4f (Exact: 24/7 = %.4f)\n', x_star(1), 24/7);
fprintf('Optimal x2* : %.4f (Exact:  5/7 = %.4f)\n', x_star(2), 5/7);
fprintf('Minimum f*  : %.4f (Exact: -125/7 = %.4f)\n\n', fval, -125/7);

fprintf('--- Lagrange Multipliers (Dual Variables) ---\n');
fprintf('mu_1 (inactive) : %.4f (Exact: 0)\n', lambda.ineqlin(1));
fprintf('mu_2 (active)   : %.4f (Exact: 11/7 = %.4f)\n', lambda.ineqlin(2), 11/7);
fprintf('mu_3 (active)   : %.4f (Exact:  2/7 = %.4f)\n', lambda.ineqlin(3), 2/7);


%% 2. Plot the Feasible Region and Verification
figure('Color', 'w', 'Position', [100, 100, 750, 600]);
hold on; grid on; box on;

% Vertices of the triangular feasible region
% A = (0, 11), B = (0, -1), C = (24/7, 5/7)
v_x1 = [0, 0, 24/7];
v_x2 = [11, -1, 5/7];

% Fill feasible region
fill(v_x1, v_x2, [0.8 0.9 1], 'FaceAlpha', 0.6, 'EdgeColor', 'none', ...
    'DisplayName', 'Feasible Region');

% Boundary Lines
x1_plot = linspace(-0.5, 4.5, 200);
plot([0, 0], [-2, 13], 'm-', 'LineWidth', 2, 'DisplayName', 'g_1: -x_1 \le 0');
plot(x1_plot, 11 - 3*x1_plot, 'r-', 'LineWidth', 2, 'DisplayName', 'g_2: 3x_1 + x_2 \le 11');
plot(x1_plot, 0.5*x1_plot - 1, 'b-', 'LineWidth', 2, 'DisplayName', 'g_3: x_1 - 2x_2 \le 2');

% Plot Optimal Point
plot(x_star(1), x_star(2), 'kp', 'MarkerFaceColor', 'y', 'MarkerSize', 15, ...
    'LineWidth', 1.5, 'DisplayName', sprintf('x^* = (24/7, 5/7)'));

% Plot Negative Objective Gradient (-grad f points in direction of descent)
scale = 0.4;
quiver(x_star(1), x_star(2), -f(1)*scale, -f(2)*scale, 0, 'Color', [0.85 0.33 0.1], ...
    'LineWidth', 2, 'MaxHeadSize', 0.5, 'DisplayName', 'Descent Dir: -\nabla f');

% Plot Active Constraint Outward Gradients
quiver(x_star(1), x_star(2), A(2,1)*scale, A(2,2)*scale, 0, 'Color', 'r', ...
    'LineWidth', 2, 'MaxHeadSize', 0.5, 'DisplayName', '\nabla g_2 = [3, 1]^T');
quiver(x_star(1), x_star(2), A(3,1)*scale, A(3,2)*scale, 0, 'Color', 'b', ...
    'LineWidth', 2, 'MaxHeadSize', 0.5, 'DisplayName', '\nabla g_3 = [1, -2]^T');

% Formatting
xlabel('x_1', 'FontSize', 12);
ylabel('x_2', 'FontSize', 12);
title('LP Feasible Region and KKT Optimality Verification', 'FontSize', 13);
xlim([-0.5, 4.5]);
ylim([-2, 12]);
legend('Location', 'northeast', 'FontSize', 10);
hold off;