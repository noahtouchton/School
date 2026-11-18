function optimal_nose_cone_p2()
    clear; clc; close all;

    fprintf('=== Part (b) & (e): Analytical Nose Shapes ===\n');
    % Different initial slopes u0 determine different a/l fineness ratios
    u0_values = [0.2, 0.5, 1.0, 2.0, 5.0];
    
    figure('Name', 'Nose Cone Profiles (Analytical)');
    hold on;
    
    for k = 1:length(u0_values)
        u0 = u0_values(k);
        
        % Generate range of slope u from u0 down to tip u(l) = 1
        u = linspace(u0, 1.0, 500);
        
        % Parametric equation for r / r(l)
        r_ratio = (1 + u.^2).^2 ./ (4 * u.^3);
        
        % Parametric equation for (l - x) / r(l)
        l_minus_x_ratio = 0.25 * (3./(4*u.^4) + 1./u.^2 - 7/4 + log(u));
        
        % At u = u0, x = 0, so l / r(l) is:
        l_over_rl = 0.25 * (3./(4*u0^4) + 1./u0^2 - 7/4 + log(u0));
        
        % Normalize to x/a and r/a where a = r(0)
        a_over_rl = (1 + u0^2)^2 / (4 * u0^3);
        
        r_normalized = r_ratio / a_over_rl;             % r / a
        x_normalized = (l_over_rl - l_minus_x_ratio) / a_over_rl; % x / a
        
        % Calculate fineness ratio a / l
        a_over_l = 1 / l_over_rl;
        
        plot(x_normalized, r_normalized, 'LineWidth', 1.5, ...
            'DisplayName', sprintf('a/l = %.3f (u_0 = %.1f)', a_over_l, u0));
        plot(x_normalized, -r_normalized, 'LineWidth', 1.5, 'HandleVisibility', 'off'); % mirror image
    end
    
    xlabel('x / a'); ylabel('r / a');
    title('Analytical Optimal Nose Cone Shapes');
    legend('Location', 'southeast');
    grid on; axis equal;
    xlim([0 4]); ylim([-1.5 1.5]);

    fprintf('\n=== Part (c): Limiting Values as a/l -> 0 ===\n');
    fprintf('As a/l -> 0 (slender body limit, u0 -> 0):\n');
    fprintf('- The nose shape becomes infinitely long and needle-like.\n');
    fprintf('- The drag coefficient C_D approaches 0 as the fineness ratio vanishes.\n');

    fprintf('\n=== Part (d) & (e): Numerical Solution vs Analytical ===\n');
    % Solve numerically using gradient descent (adapted from Problem 1)
    l_span = 1.0; 
    N = 500;
    dx = l_span / N;
    x = linspace(0, l_span, N+1);
    
    tol = 1e-6;
    kappa = 0.05;  % Learning rate
    max_iter = 1000;
    
    % Initial guess for control u(x)
    u_num = 1.5 * ones(1, N+1);
    
    for iter = 1:max_iter
        % 1. Forward Integration for states: r(x) and J(x)
        r = zeros(1, N+1);
        J_cost = zeros(1, N+1);
        r(1) = 0.5; % initial radius a
        J_cost(1) = 0.0;
        
        for i = 1:N
            % dr/dx = -u
            r(i+1) = r(i) + dx * (-u_num(i));
            % dJ/dx = 4 r u^3 / (1 + u^2)
            dJ = 4 * r(i) * u_num(i)^3 / (1 + u_num(i)^2);
            J_cost(i+1) = J_cost(i) + dx * dJ;
        end
        
        % Terminal cost phi = 2*r(l)^2 + J(l)
        total_cost = 2 * r(end)^2 + J_cost(end);
        
        % 2. Backward Integration for costates
        lambda_r = zeros(1, N+1);
        lambda_r(end) = 4 * r(end); % dphi/dr at x = l
        
        for i = N:-1:1
            dr_val = 4 * u_num(i+1)^3 / (1 + u_num(i+1)^2);
            lambda_r(i) = lambda_r(i+1) + dx * dr_val;
        end
        
        % 3. Evaluate gradient H_u
        Hu = zeros(1, N+1);
        for i = 1:N+1
            fu_r = -1.0; 
            fu_J = 4 * r(i) * (3*u_num(i)^2 * (1+u_num(i)^2) - u_num(i)^3 * (2*u_num(i))) / (1+u_num(i)^2)^2;
            Hu(i) = lambda_r(i) * fu_r + 1.0 * fu_J;
        end
        
        delta_u = -kappa * Hu;
        delta_u_aver = sqrt((1/l_span) * trapz(x, delta_u .* delta_u));
        
        if delta_u_aver < tol
            fprintf('Numerical solver converged at iteration %d! Final C_D = %.6f\n', iter, total_cost);
            break;
        end
        
        u_num = u_num + delta_u;
        u_num = max(u_num, 0.05); 
    end
    
    % Plot numerical vs analytical comparison (Part e)
    figure('Name', 'Numerical vs Analytical Comparison');
    plot(x, r, 'ro-', 'LineWidth', 1.5, 'DisplayName', 'Numerical Solution');
    hold on;
    grid on;
    xlabel('x'); ylabel('r(x)');
    title('Comparison of Numerical Optimal Control vs. Analytical Profile');
    legend();
end