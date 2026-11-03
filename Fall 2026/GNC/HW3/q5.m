% Clear workspace and close figures
clear; clc; close('all');

% Setup grid
[X1, X2] = meshgrid(linspace(-0.5, 1.5, 400), linspace(-0.5, 1.5, 400));

% Constraints
g1 = X2 - (1 - X1).^3;
g2 = -X2;

% Feasible region mask (g1 <= 0 and g2 <= 0)
feasible = (g1 <= 0) & (g2 <= 0);

figure('Color', 'w', 'Position', [100, 100, 800, 650]);
hold on; grid on; box on;

% 1. Plot the feasible region
% Shaded region
contourf(X1, X2, double(feasible), [1 1], 'FaceColor', [0.8 0.9 1], ...
    'FaceAlpha', 0.6, 'EdgeColor', 'none', 'DisplayName', 'Feasible Region');

% Boundary curves
fimplicit(@(x1, x2) x2 - (1 - x1).^3, [-0.5 1.5 -0.5 1.5], 'r-', ...
    'LineWidth', 2, 'DisplayName', 'g_1(x) = x_2 - (1-x_1)^3 = 0');
yline(0, 'b-', 'LineWidth', 2, 'DisplayName', 'g_2(x) = -x_2 = 0');

% 2. Plot optimal point x* = (1, 0)
x_star = [1, 0];
plot(x_star(1), x_star(2), 'kp', 'MarkerFaceColor', 'y', 'MarkerSize', 14, ...
    'LineWidth', 1.5, 'DisplayName', 'Optimal Point x^* = (1,0)');

% 3. Plot Gradients as Vectors using quiver
scale = 0.4; % Visual scaling for arrows

% - \nabla f = [-1; 0] (gradient of function to be minimized)
quiver(x_star(1), x_star(2), -1*scale, 0*scale, 0, 'Color', [0.85 0.33 0.1], ...
    'LineWidth', 2.5, 'MaxHeadSize', 0.5, 'DisplayName', '\nabla f(x^*) = [-1, 0]^T');

% - \nabla g1 = [0; 1]
quiver(x_star(1), x_star(2), 0*scale, 1*scale, 0, 'Color', 'r', ...
    'LineWidth', 2.5, 'MaxHeadSize', 0.5, 'DisplayName', '\nabla g_1(x^*) = [0, 1]^T');

% - \nabla g2 = [0; -1]
quiver(x_star(1), x_star(2), 0*scale, -1*scale, 0, 'Color', 'b', ...
    'LineWidth', 2.5, 'MaxHeadSize', 0.5, 'DisplayName', '\nabla g_2(x^*) = [0, -1]^T');

% Plot styling
xlabel('x_1', 'FontSize', 12);
ylabel('x_2', 'FontSize', 12);
title('Feasible Region, Optimal Point, and Gradients', 'FontSize', 14);
xlim([-0.2, 1.4]);
ylim([-0.4, 1.2]);
legend('Location', 'northwest', 'FontSize', 10);
axis equal;
hold off;