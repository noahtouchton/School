function problem5_zermelo_navigation()
clear; clc; close all;
alpha = 0.2;

% Solve for [theta0, tf] such that x(tf) = 0 and y(tf) = 0
% Initial guess: heading roughly toward origin (-3*pi/4), tf ~ 1.5
guess = [-2.4, 1.5];
options = optimoptions('fsolve', 'Display', 'iter', 'TolFun', 1e-10, 'TolX', 1e-10);
sol = fsolve(@(vars) shooting_residual(vars, alpha), guess, options);

theta0_opt = sol(1);
tf_opt = sol(2);

% Integrate optimal trajectory with converged parameters
[t, state] = ode45(@(t, s) zermelo_ode(t, s, alpha), [0, tf_opt], [1; 1; theta0_opt]);
x = state(:, 1);
y = state(:, 2);
theta = state(:, 3);

% Verify Hint Condition: sec(theta(0)) - 2*alpha == sec(theta(tf))
lhs = sec(theta(1)) - 2 * alpha;
rhs = sec(theta(end));

fprintf('\n=== Problem 5(b) Results (alpha = %.2f) ===\n', alpha);
fprintf('Optimal Initial Heading theta(0): %.6f rad (%.2f deg)\n', theta0_opt, rad2deg(theta0_opt));
fprintf('Optimal Final Heading theta(tf):  %.6f rad (%.2f deg)\n', theta(end), rad2deg(theta(end)));
fprintf('Minimum Time tf:                  %.6f\n', tf_opt);
fprintf('Terminal State:                   x(tf) = %.2e, y(tf) = %.2e\n', x(end), y(end));
fprintf('Hint Check: sec(theta0) - 2*alpha = %.6f | sec(theta_tf) = %.6f\n', lhs, rhs);

% Plot Optimal Trajectory in x-y Plane with Current Field
figure('Name', 'Problem 5(b): Minimum-Time Path');
[X_grid, Y_grid] = meshgrid(linspace(-0.2, 1.2, 15), linspace(-0.2, 1.2, 15));
U_flow = -alpha * (3 * Y_grid - Y_grid.^3);
V_flow = zeros(size(X_grid));
quiver(X_grid, Y_grid, U_flow, V_flow, 'Color', [0.7 0.7 0.7], 'DisplayName', 'Current u(y)');
hold on;

plot(x, y, 'b-', 'LineWidth', 2, 'DisplayName', 'Optimal Trajectory');
plot(1, 1, 'go', 'MarkerFaceColor', 'g', 'MarkerSize', 8, 'DisplayName', 'Start (1,1)');
plot(0, 0, 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 8, 'DisplayName', 'Origin (0,0)');
xlabel('x'); ylabel('y');
title(sprintf('Zermelo Minimum-Time Path (\\alpha = %.1f, t_f = %.4f)', alpha, tf_opt));
legend('Location', 'best'); grid on; axis equal;
end

function res = shooting_residual(vars, alpha)
theta0 = vars(1);
tf = vars(2);
[~, state] = ode45(@(t, s) zermelo_ode(t, s, alpha), [0, tf], [1; 1; theta0]);
res = [state(end, 1); state(end, 2)]; % Target: x(tf) = 0, y(tf) = 0
end

function dsdt = zermelo_ode(~, s, alpha)
y = s(2);
theta = s(3);
u_y = -alpha * (3*y - y^3);
dxdt = cos(theta) + u_y;
dydt = sin(theta);
dthetadt = 3 * alpha * (1 - y^2) * cos(theta)^2;
dsdt = [dxdt; dydt; dthetadt];
end