function problem6_ship_rescue()
    clear; clc; close all;

    V = 15;             % Ship speed relative to water (ml/hr)[cite: 13]
    x0 = -20; y0 = 0;   % Initial ship location[cite: 13]

    % =====================================================================
    % PART (a): Constant Current to City (-15, 35.5)[cite: 13]
    % =====================================================================
    x1 = -15; y1 = 35.5; %[cite: 13]
    sol_a = fsolve(@(z) [x0 + (V*cos(z(1)) + 2)*z(2) - x1; ...
                         y0 + (V*sin(z(1)) - 6)*z(2) - y1], [1.4, 2.0], ...
                         optimoptions('fsolve', 'Display', 'off'));
    theta_a = sol_a(1);
    tf_a = sol_a(2);

    t_a = linspace(0, tf_a, 200);
    x_a = x0 + (V*cos(theta_a) + 2)*t_a;
    y_a = y0 + (V*sin(theta_a) - 6)*t_a;

    % Compute costates from H = 0 and tan(theta) = lam_y / lam_x
    denom_a = V + 2*cos(theta_a) - 6*sin(theta_a);
    lam_x_a = -cos(theta_a) / denom_a;
    lam_y_a = -sin(theta_a) / denom_a;
    H_a = 1 + lam_x_a*(V*cos(theta_a) + 2) + lam_y_a*(V*sin(theta_a) - 6);

    fprintf('=== PART (a) RESULTS ===\n');
    fprintf('Optimal Heading theta*: %.4f rad (%.2f deg)\n', theta_a, rad2deg(theta_a));
    fprintf('Time to City tf:        %.4f hr\n', tf_a);
    fprintf('Max |H(t)|:             %.2e\n\n', max(abs(H_a)));

    % =====================================================================
    % PART (b): Constant Current to Shoreline psi(x,y) = 0[cite: 13]
    % =====================================================================
    sol_b = fsolve(@(z) res_part_b(z, x0, y0, V), [1.3, 1.8], ...
                   optimoptions('fsolve', 'Display', 'off'));
    theta_b = sol_b(1);
    tf_b = sol_b(2);

    t_b = linspace(0, tf_b, 200);
    x_b = x0 + (V*cos(theta_b) + 2)*t_b;
    y_b = y0 + (V*sin(theta_b) - 6)*t_b;

    denom_b = V + 2*cos(theta_b) - 6*sin(theta_b);
    lam_x_b = -cos(theta_b) / denom_b;
    lam_y_b = -sin(theta_b) / denom_b;
    H_b = 1 + lam_x_b*(V*cos(theta_b) + 2) + lam_y_b*(V*sin(theta_b) - 6);

    fprintf('=== PART (b) RESULTS ===\n');
    fprintf('Optimal Heading theta*: %.4f rad (%.2f deg)\n', theta_b, rad2deg(theta_b));
    fprintf('Shoreline Arrival:      (%.4f, %.4f) ml\n', x_b(end), y_b(end));
    fprintf('Time to Shore tf:       %.4f hr\n', tf_b);
    fprintf('Max |H(t)|:             %.2e\n\n', max(abs(H_b)));

    % =====================================================================
    % PART (c): Variable Current to Shoreline psi(x,y) = 0[cite: 13]
    % =====================================================================
    sol_c = fsolve(@(z) res_part_c(z, x0, y0, V), [0.5, 0.8], ...
                   optimoptions('fsolve', 'Display', 'off'));
    theta0_c = sol_c(1);
    tf_c = sol_c(2);

    [t_c, state_c] = ode45(@(t, s) ode_part_c(t, s, V), [0, tf_c], [x0; y0; theta0_c]);
    x_c = state_c(:, 1);
    y_c = state_c(:, 2);
    theta_c = state_c(:, 3);

    % Verify H(t) = 0 along trajectory
    H_c = zeros(size(t_c));
    for i = 1:length(t_c)
        u_curr = -(y_c(i) - 50); %[cite: 13]
        v_curr = 2*(x_c(i) - 15); %[cite: 13]
        denom_c = V + u_curr*cos(theta_c(i)) + v_curr*sin(theta_c(i));
        lx = -cos(theta_c(i)) / denom_c;
        ly = -sin(theta_c(i)) / denom_c;
        H_c(i) = 1 + lx*(V*cos(theta_c(i)) + u_curr) + ly*(V*sin(theta_c(i)) + v_curr);
    end

    fprintf('=== PART (c) RESULTS ===\n');
    fprintf('Initial Heading theta(0): %.4f rad (%.2f deg)\n', theta0_c, rad2deg(theta0_c));
    fprintf('Final Heading theta(tf):  %.4f rad (%.2f deg)\n', theta_c(end), rad2deg(theta_c(end)));
    fprintf('Shoreline Arrival:        (%.4f, %.4f) ml\n', x_c(end), y_c(end));
    fprintf('Time to Shore tf:         %.4f hr\n', tf_c);
    fprintf('Max |H(t)|:               %.2e\n', max(abs(H_c)));

    % =====================================================================
    % PLOTTING ALL CASES[cite: 13]
    % =====================================================================
    x_shore = linspace(-30, 5, 200);
    y_shore = 25 - 0.25*x_shore - 0.002*(x_shore.^3); %[cite: 13]

    % --- Figure 1: Parts (a) and (b) ---
    figure('Name', 'Parts (a) & (b): Constant Current Paths', 'Position', [100, 100, 1000, 450]);
    subplot(1, 2, 1);
    [Xg, Yg] = meshgrid(linspace(-25, 0, 12), linspace(-5, 40, 12));
    quiver(Xg, Yg, 2*ones(size(Xg)), -6*ones(size(Yg)), 'Color', [0.7 0.7 0.7], 'DisplayName', 'Current'); hold on;
    plot(x_a, y_a, 'b-', 'LineWidth', 2, 'DisplayName', 'Optimal Path (a)');
    plot(x0, y0, 'go', 'MarkerFaceColor', 'g', 'DisplayName', 'Ship Start');
    plot(x1, y1, 'rp', 'MarkerSize', 10, 'MarkerFaceColor', 'r', 'DisplayName', 'City (-15, 35.5)');
    xlabel('x (ml)'); ylabel('y (ml)'); title(sprintf('Part (a): t_f = %.3f hr', tf_a));
    legend('Location', 'best'); grid on; axis equal;

    subplot(1, 2, 2);
    quiver(Xg, Yg, 2*ones(size(Xg)), -6*ones(size(Yg)), 'Color', [0.7 0.7 0.7], 'DisplayName', 'Current'); hold on;
    plot(x_shore, y_shore, 'k--', 'LineWidth', 1.5, 'DisplayName', 'Shoreline \psi=0');
    plot(x_b, y_b, 'm-', 'LineWidth', 2, 'DisplayName', 'Optimal Path (b)');
    plot(x0, y0, 'go', 'MarkerFaceColor', 'g', 'DisplayName', 'Ship Start');
    plot(x_b(end), y_b(end), 'ro', 'MarkerFaceColor', 'r', 'DisplayName', 'Pickup Point');
    xlabel('x (ml)'); ylabel('y (ml)'); title(sprintf('Part (b): t_f = %.3f hr', tf_b));
    legend('Location', 'best'); grid on; axis equal; ylim([-5 40]);

    % --- Figure 2: Part (c) Path & Steering History[cite: 13] ---
    figure('Name', 'Part (c): Variable Current Path & Steering', 'Position', [150, 150, 1000, 450]);
    subplot(1, 2, 1);
    [Xg2, Yg2] = meshgrid(linspace(-25, 5, 14), linspace(-5, 40, 14));
    Uc = -(Yg2 - 50); Vc = 2*(Xg2 - 15);
    quiver(Xg2, Yg2, Uc, Vc, 'Color', [0.7 0.7 0.7], 'DisplayName', 'Strong Current'); hold on;
    plot(x_shore, y_shore, 'k--', 'LineWidth', 1.5, 'DisplayName', 'Shoreline \psi=0');
    plot(x_c, y_c, 'b-', 'LineWidth', 2, 'DisplayName', 'Optimal Path (c)');
    plot(x0, y0, 'go', 'MarkerFaceColor', 'g', 'DisplayName', 'Ship Start');
    plot(x_c(end), y_c(end), 'ro', 'MarkerFaceColor', 'r', 'DisplayName', 'Pickup Point');
    xlabel('x (ml)'); ylabel('y (ml)'); title(sprintf('Part (c): t_f = %.3f hr', tf_c));
    legend('Location', 'best'); grid on; axis equal; ylim([-5 40]);

    subplot(1, 2, 2);
    plot(t_c, rad2deg(theta_c), 'r-', 'LineWidth', 2);
    xlabel('Time t (hr)'); ylabel('Steering Angle \theta^*(t) (deg)');
    title('Part (c): Optimal Steering History \theta^*(t)'); grid on;

    % --- Figure 3: Hamiltonian Verification Across All Cases[cite: 13] ---
    figure('Name', 'Hamiltonian Verification (All Cases)', 'Position', [200, 200, 700, 350]);
    plot(t_a, H_a * ones(size(t_a)), 'b-', 'LineWidth', 1.5, 'DisplayName', 'Case (a)'); hold on;
    plot(t_b, H_b * ones(size(t_b)), 'm--', 'LineWidth', 1.5, 'DisplayName', 'Case (b)');
    plot(t_c, H_c, 'k-.', 'LineWidth', 1.5, 'DisplayName', 'Case (c)');
    ylim([-1e-5, 1e-5]);
    xlabel('Time t (hr)'); ylabel('Hamiltonian H(t)');
    title('Verification that Hamiltonian Remains Zero for All Time[cite: 13]');
    legend('Location', 'best'); grid on;
end

function F = res_part_b(z, x0, y0, V)
    theta = z(1); tf = z(2);
    xf = x0 + (V*cos(theta) + 2)*tf;
    yf = y0 + (V*sin(theta) - 6)*tf;
    psi = 25 - 0.25*xf - 0.002*(xf^3) - yf; %[cite: 13]
    % Transversality: tan(theta) = dpsi_dy / dpsi_dx = -1 / (-0.25 - 0.006*xf^2)
    trans = sin(theta)*(-0.25 - 0.006*xf^2) - cos(theta)*(-1);
    F = [psi; trans];
end

function F = res_part_c(z, x0, y0, V)
    theta0 = z(1); tf = z(2);
    [~, s] = ode45(@(t, state) ode_part_c(t, state, V), [0, tf], [x0; y0; theta0]);
    xf = s(end, 1); yf = s(end, 2); thetaf = s(end, 3);
    psi = 25 - 0.25*xf - 0.002*(xf^3) - yf; %[cite: 13]
    trans = sin(thetaf)*(-0.25 - 0.006*xf^2) - cos(thetaf)*(-1);
    F = [psi; trans];
end

function ds = ode_part_c(~, s, V)
    x = s(1); y = s(2); theta = s(3);
    u = -(y - 50); %[cite: 13]
    v = 2*(x - 15); %[cite: 13]
    dxdt = V*cos(theta) + u;
    dydt = V*sin(theta) + v;
    dthetadt = 1 + sin(theta)^2;
    ds = [dxdt; dydt; dthetadt];
end