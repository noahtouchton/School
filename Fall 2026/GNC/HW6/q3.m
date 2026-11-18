function problem4_plots_and_verification()
    clear; clc; close all;

    % Define integration constants [c1, c2] and final times tf for Cases (a)-(e)
    cases = {
        'Case (a): Fixed End [1, 2]',       [ 0.681091, -0.264246], 3.0;
        'Case (b): x1(3)=1, x2(3) Free',    [-0.625651,  0.031149], 3.0;
        'Case (c): Soft Penalty [1, 2]',    [-0.161458, -0.055978], 3.0;
        'Case (d): Line 2x1+5x2=20',        [-1.434134, -0.107102], 3.0;
        'Case (e): Free Time Moving Line',  [-1.783920, -0.183392], 2.309045
    };

    num_cases = size(cases, 1);

    % Figures for Sub-part (c) and (d)
    fig_time = figure('Name', 'Sub-part (c): Trajectories vs. Time', 'Position', [100, 100, 1000, 700]);
    fig_phase = figure('Name', 'Sub-part (c): Phase Plane (x1 vs x2)', 'Position', [150, 150, 700, 600]);
    fig_ham = figure('Name', 'Sub-part (d): Hamiltonian Verification', 'Position', [200, 200, 800, 400]);

    colors = lines(num_cases);

    fprintf('========================================================================\n');
    fprintf('PROBLEM 4: BOUNDARY CHECKS, HAMILTONIAN & OPTIMAL COSTS\n');
    fprintf('========================================================================\n');

    for k = 1:num_cases
        label = cases{k, 1};
        c1 = cases{k, 2}(1);
        c2 = cases{k, 2}(2);
        tf = cases{k, 3};

        t = linspace(0, tf, 500);

        % Analytical expressions
        lam1 = c1 * ones(size(t));
        lam2 = c1 + c2 * exp(t);
        u    = -lam2;
        x1   = c1 * (1 - t - exp(-t)) + c2 * (1 - cosh(t));
        x2   = -c1 * (1 - exp(-t)) - c2 * sinh(t);

        % Hamiltonian: H = 0.5*u^2 + lam1*x2 + lam2*(-x2 + u)
        H = 0.5 * (u.^2) + lam1 .* x2 + lam2 .* (-x2 + u);

        % Optimal Cost: J = 0.5 * int_0^tf u^2 dt
        J_opt = 0.5 * trapz(t, u.^2);

        % Print Verification Summary
        fprintf('\n%s (tf = %.4f)\n', label, tf);
        fprintf('  Terminal State: x1(tf) = %.4f, x2(tf) = %.4f\n', x1(end), x2(end));
        fprintf('  Hamiltonian H:  Mean = %.6f (Max deviation = %.2e)\n', mean(H), max(abs(H - mean(H))));
        fprintf('  Optimal Cost J: %.6f\n', J_opt);

        % ---------------------------------------------------------
        % Sub-part (c): Plot Trajectories vs. Time
        % ---------------------------------------------------------
        figure(fig_time);
        subplot(3, 1, 1); hold on;
        plot(t, x1, 'Color', colors(k,:), 'LineWidth', 1.5, 'DisplayName', label);
        ylabel('x_1(t)'); title('State x_1(t) vs. Time'); grid on;

        subplot(3, 1, 2); hold on;
        plot(t, x2, 'Color', colors(k,:), 'LineWidth', 1.5, 'DisplayName', label);
        ylabel('x_2(t)'); title('State x_2(t) vs. Time'); grid on;

        subplot(3, 1, 3); hold on;
        plot(t, u, 'Color', colors(k,:), 'LineWidth', 1.5, 'DisplayName', label);
        xlabel('Time t'); ylabel('u^*(t)'); title('Optimal Control u^*(t) vs. Time'); grid on;

        % ---------------------------------------------------------
        % Sub-part (c): Plot Phase Plane (x1 - x2)
        % ---------------------------------------------------------
        figure(fig_phase); hold on;
        plot(x1, x2, 'Color', colors(k,:), 'LineWidth', 1.5, 'DisplayName', label);
        plot(x1(end), x2(end), 'o', 'Color', colors(k,:), 'MarkerFaceColor', colors(k,:), 'HandleVisibility', 'off');

        % ---------------------------------------------------------
        % Sub-part (d): Plot Hamiltonian vs. Time
        % ---------------------------------------------------------
        figure(fig_ham); hold on;
        plot(t, H, 'Color', colors(k,:), 'LineWidth', 1.5, 'DisplayName', label);
    end

    % Finalize Time Plots formatting
    figure(fig_time);
    subplot(3, 1, 1); legend('Location', 'northwest');

    % Finalize Phase Plane formatting with target constraints
    figure(fig_phase);
    plot(1, 2, 'kp', 'MarkerSize', 12, 'MarkerFaceColor', 'y', 'DisplayName', 'Target [1, 2]');
    x1_line = linspace(-1, 6, 100);
    plot(x1_line, (20 - 2*x1_line)/5, 'k--', 'LineWidth', 1.2, 'DisplayName', '2x_1 + 5x_2 = 20');
    xlabel('State x_1'); ylabel('State x_2');
    title('Optimal Paths in the x_1 - x_2 Plane');
    legend('Location', 'northwest'); grid on;

    % Finalize Hamiltonian Plot formatting
    figure(fig_ham);
    xlabel('Time t'); ylabel('Hamiltonian H(t)');
    title('Hamiltonian Constancy Verification Along Optimal Trajectories');
    legend('Location', 'best'); grid on;
end