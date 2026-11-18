clc; clear; close all;

N = 1000;

t0 = 0.0;
x0 = 1.0;
tf = 1.0;
dt = tf/N;

tol = 1e-5;
kappa = 0.2; % k<0 for max k>0 for min


t = linspace(t0, tf, N+1);

maxtries = 500;

u = zeros(1, N+1);

for iter = 1:maxtries
    x = zeros(1, N+1);
    x(1) = x0;


    for i = 1:N
        f_val = system_dynamics(x(i), u(i), t(i));
        x(i+1) = x(i) + dt*f_val;
    end
    
    cost_val = cost(x(end));
    %lambda^T = dphi/dx(tf)
    
    lambda = zeros(1, N+1);
    lambda(end) = cost_x(x(end));

    % compute the rest of the lambda values using backwards integration
    for i = N: -1 : 1
        % lambda dot^T = -lambda^T * f_x(x,u,t)
        fx = fx_dynamics(x(i+1), u(i+1), t(i+1));
        lambda(i) = lambda(i+1) + dt*lambda(i+1)*fx;
    end

    %evaluate H_u = lambda^T * f_u
    Hu = zeros(1,N+1);
    for i = 1:N+1
        fu = fu_dynamics(x(i), u(i), t(i));
        Hu(i) = lambda(i)*fu;
    end

    delta_u = -kappa*Hu;
    integral_val = trapz(t, delta_u .* delta_u);
    delta_u_aver = sqrt((1/tf) * integral_val);

    if delta_u_aver < tol
        fprintf('Converged at iteration %d! Delta u aver = %.6f\n', iter, delta_u_aver);
        break;
    end
    u = u + delta_u;

    if mod(iter, 50) == 0 
        fprintf('Iteration: %d, Cost = %.6f, Delta u aver = %.6f\n', iter, cost_val, delta_u_aver);
    end
end

% ==========================================
% Plotting Results
% ==========================================
figure;
subplot(3,1,1);
plot(t, x, 'LineWidth', 1.5);
xlabel('Time t'); ylabel('State x(t)');
title('Optimal State Trajectory'); grid on;

subplot(3,1,2);
plot(t, u, 'r', 'LineWidth', 1.5);
xlabel('Time t'); ylabel('Control u(t)');
title('Optimal Control'); grid on;

subplot(3,1,3);
plot(t, Hu, 'k', 'LineWidth', 1.5);
xlabel('Time t'); ylabel('H_u(t)');
title('Gradient / Hamiltonian Derivative wrt u'); grid on;

function f = system_dynamics(x, u, t)
% x_dot = -x + u
    f = -x + u;
end

function fx = fx_dynamics(x, u, t)
    % derivative with respect to x of system dynamics
    fx = -1;
end

function fu = fu_dynamics(x, u, t)
    % derivative with respect to u of system dynamics
    fu =1;
end


function phi = cost(x)
    %arbitrary cost function. only final result penalty
    phi = 1/2 * x^2;
end
function dphi = cost_x(x)
    %derivative with respect to x of phi
    dphi = x;
end