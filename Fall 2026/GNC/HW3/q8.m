% Clear workspace and close figures
clear; clc; close('all');

%% 1. Aircraft Physical Parameters & Scaling
W = 180000;              % Weight (lbf)
rho = 0.002203;          % Air density at sea level (slugs/ft^3)
S = 1560;                % Wing reference area (ft^2)
g = 32.174;              % Gravity (ft/s^2)
eps = 0.0349;            % Engine alignment angle (rad)

ell = 2 * W / (rho * g * S);   % Reference length scale (ft)
V_scale = sqrt(g * ell);       % Reference velocity scale (ft/s)

fprintf('--- Reference Parameters ---\n');
fprintf('ell     = %.2f ft\n', ell);
fprintf('V_scale = %.2f ft/s\n\n', V_scale);

%% 2. Setup Optimization with fmincon
% State vector: x = [alpha (rad), v_nd (dimensionless), gamma (rad)]
% Objective: minimize -V_scale * v_nd * sin(gamma)
obj_fun = @(x) -V_scale * x(2) * sin(x(3));

% Initial guess: alpha = 6 deg, V = 340 ft/s, gamma = 6 deg
x0 = [deg2rad(6); 340 / V_scale; deg2rad(6)];

% Variable bounds
lb = [deg2rad(0); 100 / V_scale; deg2rad(0)];
ub = [deg2rad(15); 600 / V_scale; deg2rad(20)];

% Non-linear constraints function handle
nonlcon = @(x) aircraft_constraints(x, eps);

% Solver options
options = optimoptions('fmincon', ...
    'Display', 'iter-detailed', ...
    'Algorithm', 'sqp', ...
    'StepTolerance', 1e-12, ...
    'ConstraintTolerance', 1e-10, ...
    'OptimalityTolerance', 1e-10);

% Solve
[x_opt, fval, exitflag, output, lambda] = fmincon(obj_fun, x0, [], [], [], [], lb, ub, nonlcon, options);

%% 3. Display Results for Part (b)
alpha_opt_deg = rad2deg(x_opt(1));
V_opt_fps     = x_opt(2) * V_scale;
gamma_opt_deg = rad2deg(x_opt(3));
max_roc       = -fval;

fprintf('\n==============================================\n');
fprintf('Part (b) Results:\n');
fprintf('----------------------------------------------\n');
fprintf('Maximum Rate of Climb (ROC) : %6.2f ft/sec (Expected: ~37.6 ft/sec)\n', max_roc);
fprintf('True Airspeed (V)           : %6.2f ft/sec (Expected: ~342 ft/sec)\n', V_opt_fps);
fprintf('Angle of Attack (alpha)     : %6.2f deg    (Expected: ~6.39 deg)\n', alpha_opt_deg);
fprintf('Flight Path Angle (gamma)   : %6.2f deg    (Expected: ~6.31 deg)\n', gamma_opt_deg);
fprintf('==============================================\n\n');

%% 4. Verify Necessary Conditions for Part (c)
% Multipliers for equality constraints
lambda_eq = lambda.eqnonlin;

% Analytical gradients of the objective and constraints at x_opt
[grad_f, Jac_h] = evaluate_derivatives(x_opt, eps, V_scale);

% Gradient of the Lagrangian: grad_L = grad_f + Jac_h^T * lambda_eq
% (Note: fmincon defines Lagrangian as f + sum(lambda * h))
grad_L = grad_f + Jac_h' * lambda_eq;

fprintf('Part (c) FONC Verification:\n');
fprintf('----------------------------------------------\n');
fprintf('Lagrange Multiplier lambda_1 (Parallel) : %+12.6e\n', lambda_eq(1));
fprintf('Lagrange Multiplier lambda_2 (Normal)   : %+12.6e\n\n', lambda_eq(2));
fprintf('Gradient of the Lagrangian at x^* (dL/dx):\n');
fprintf('  dL/d(alpha) = %+12.6e\n', grad_L(1));
fprintf('  dL/d(v_nd)  = %+12.6e\n', grad_L(2));
fprintf('  dL/d(gamma) = %+12.6e\n', grad_L(3));
fprintf('Infinity-Norm ||grad_L||_inf             = %12.6e\n', norm(grad_L, inf));

if norm(grad_L, inf) < 1e-4
    fprintf('\n-> Verified: First-order necessary optimality conditions hold (|grad_L| ~ 0).\n');
else
    fprintf('\n-> Note: Bound constraints or tolerance thresholds active.\n');
end


%% --- Helper Functions ---
function [c, ceq] = aircraft_constraints(x, eps)
    c = []; % No inequality constraints
    
    alpha = x(1);
    v     = x(2);
    gamma = x(3);
    
    % Aerodynamic Models (normalized by Weight W)
    T = 0.2476 - 0.04312*v + 0.008392*v^2;
    D = (0.07351 - 0.08617*alpha + 1.996*alpha^2) * v^2;
    stall_term = max(0, alpha - 0.2094)^2;
    L = (0.1667 + 6.231*alpha - 21.65*stall_term) * v^2;
    
    % Equilibrium force equations (normalized)
    h1 = T * cos(alpha + eps) - D - sin(gamma);
    h2 = T * sin(alpha + eps) + L - cos(gamma);
    
    ceq = [h1; h2];
end

function [grad_f, Jac_h] = evaluate_derivatives(x, eps, V_scale)
    alpha = x(1);
    v     = x(2);
    gamma = x(3);
    
    % Objective f(x) = -V_scale * v * sin(gamma)
    grad_f = [ 0;
              -V_scale * sin(gamma);
              -V_scale * v * cos(gamma) ];
          
    % Functions and partial derivatives
    T = 0.2476 - 0.04312*v + 0.008392*v^2;
    dT_dv = -0.04312 + 2*0.008392*v;
    
    dD_dalpha = (-0.08617 + 2*1.996*alpha) * v^2;
    dD_dv     = 2 * (0.07351 - 0.08617*alpha + 1.996*alpha^2) * v;
    
    if alpha > 0.2094
        dL_dalpha = (6.231 - 2*21.65*(alpha - 0.2094)) * v^2;
        stall_term = (alpha - 0.2094)^2;
    else
        dL_dalpha = 6.231 * v^2;
        stall_term = 0;
    end
    dL_dv = 2 * (0.1667 + 6.231*alpha - 21.65*stall_term) * v;
    
    % Jacobian of constraints [dh1; dh2]
    Jac_h = zeros(2, 3);
    
    % dh1 / d[alpha, v, gamma]
    Jac_h(1, 1) = -T * sin(alpha + eps) - dD_dalpha;
    Jac_h(1, 2) =  dT_dv * cos(alpha + eps) - dD_dv;
    Jac_h(1, 3) = -cos(gamma);
    
    % dh2 / d[alpha, v, gamma]
    Jac_h(2, 1) =  T * cos(alpha + eps) + dL_dalpha;
    Jac_h(2, 2) =  dT_dv * sin(alpha + eps) + dL_dv;
    Jac_h(2, 3) =  sin(gamma);
end