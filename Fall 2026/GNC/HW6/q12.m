clc; clear; close all;

N = 1000;
t0 = 0.0;
r0 = 0.5;         % Initial radius a at x = 0
tf = 1.0;         % Length l of the nose cone
dt = tf/N;
tol = 1e-5;
kappa = 0.02;     % Learning rate (k > 0 for minimization)
t = linspace(t0, tf, N+1);
maxtries = 1000;

% Initial guess for control u(x) (local slope tan(theta))
u = 1.5 * ones(1, N+1);

for iter = 1:maxtries
    % Forward Integration: state r(x) and augmented cost J(x)
    r = zeros(1, N+1);
    J_cost = zeros(1, N+1);
    r(1) = r0;
    J_cost(1) = 0.0;
    
    for i = 1:N
        % dr/dx = -u[cite: 2]
        f_val = system_dynamics(r(i), u(i), t(i));
        r(i+1) = r(i) + dt * f_val;
        
        % dJ/dx = 4 * r * u^3 / (1 + u^2)[cite: 2]
        dJ = 4 * r(i) * u(i)^3 / (1 + u(i)^2);
        J_cost(i+1) = J_cost(i) + dt * dJ;
    end
    
    % Terminal cost phi = 2*r(l)^2 + J(l)[cite: 2]
    cost_val = 2 * r(end)^2 + J_cost(end);
    
    % Backward Integration for costates
    % lambda_r(tf) = dphi/dr(tf) = 4 * r(l)
    lambda = zeros(1, N+1);
    lambda(end) = 4 * r(end);
    
    for i = N : -1 : 1
        % dr_val accounts for the coupling of r in the running cost derivative
        dr_val = 4 * u(i+1)^3 / (1 + u(i+1)^2);
        lambda(i) = lambda(i+1) + dt * lambda(i+1) * 0 + dt * dr_val; 
    end
    
    % Evaluate Hu = lambda_r * f_u + d(running_cost)/du
    Hu = zeros(1, N+1);
    for i = 1:N+1
        fu_r = fu_dynamics(r(i), u(i), t(i)); % df/du = -1
        % partial derivative of running cost w.r.t u
        dJ_du = 4 * r(i) * (3*u(i)^2 * (1+u(i)^2) - u(i)^3 * (2*u(i))) / (1+u(i)^2)^2;
        Hu(i) = lambda(i) * fu_r + dJ_du;
    end
    
    delta_u = -kappa * Hu;
    integral_val = trapz(t, delta_u .* delta_u);
    delta_u_aver = sqrt((1/tf) * integral_val);
    
    if delta_u_aver < tol
        fprintf('Converged at iteration %d! Delta u aver = %.6f, Total C_D = %.6f\n', iter, delta_u_aver, cost_val);
        break;
    end
    
    u = u + delta_u;
    u = max(u, 0.01); % Keep control physically positive
    
    if mod(iter, 50) == 0 
        fprintf('Iteration: %d, C_D = %.6f, Delta u aver = %.6f\n', iter, cost_val, delta_u_aver);
    end
end

% ==========================================
% Plotting Results (Comparison format)
% ==========================================
figure('Name', 'Problem 2 Numerical Results');
subplot(3,1,1);
plot(t, r, 'LineWidth', 1.5);
xlabel('Distance x'); ylabel('Radius r(x)');
title('Optimal Nose Radius Profile r(x)'); grid on;

subplot(3,1,2);
plot(t, u, 'r', 'LineWidth', 1.5);
xlabel('Distance x'); ylabel('Control u(x) = tan(\theta)');
title('Optimal Control Profile'); grid on;

subplot(3,1,3);
plot(t, Hu, 'k', 'LineWidth', 1.5);
xlabel('Distance x'); ylabel('H_u(x)');
title('Gradient / Hamiltonian Derivative wrt u'); grid on;

% ==========================================
% Helper Functions for Problem 2
% ==========================================
function f = system_dynamics(r, u, t)
    % dr/dx = -u[cite: 2]
    f = -u;
end

function fx = fx_dynamics(r, u, t)
    % df/dr
    fx = 0.0;
end

function fu = fu_dynamics(r, u, t)
    % df/du where f = -u
    fu = -1.0;
end