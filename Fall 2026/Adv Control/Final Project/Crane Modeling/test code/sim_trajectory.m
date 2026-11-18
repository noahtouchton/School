%% sim_trajectory.m - Run the crane sim with the trajectory saved by optimize.m
% Loads trajectory.mat, drives the nonlinear dynamics (dynamics.m) with the optimal inputs,
% then lets it coast with u = 0 to show any residual swing. Compares the simulated states
% with the optimizer's nodes, checks the limits, plots and animates.
%
% u(t) is interpolated LINEARLY between nodes, because that's what trapezoidal collocation
% assumes: x(k+1) = x(k) + dt/2·(f(x_k, u_k) + f(x_k+1, u_k+1)).
clear; clc; close all;
here = fileparts(mfilename('fullpath'));
S = load(fullfile(here, 'trajectory.mat'));

t_coast = 5;      % s of u = 0 after the maneuver
animate = true;   % 3D animation at the end
fps     = 30;

%% 1. Simulate: maneuver, then coast
u_of_t = @(t) interp1(S.t, S.u.', t, 'linear', 0).';     % 3x1, zero outside [0, tf]
opts   = odeset('RelTol', 1e-9, 'AbsTol', 1e-11, 'MaxStep', S.tf/S.N/4);

[t1, X1] = ode45(@(t, x) dynamics(x, u_of_t(t)), [0 S.tf], S.x(:, 1), opts);
[t2, X2] = ode45(@(t, x) dynamics(x, zeros(3, 1)), [S.tf, S.tf + t_coast], X1(end, :).', opts);

t = [t1; t2(2:end)];   X = [X1; X2(2:end, :)].';           % X is 10 x Nt
U = [u_of_t(t1.'), zeros(3, numel(t2) - 1)];

%% 2. Checks
Xn = interp1(t, X.', S.t).';                                % sim at the optimizer's nodes
err = abs(Xn - S.x);
names = {'x', 'l', 'th', 'phi', 'psi', 'dx', 'dl', 'dth', 'dphi', 'dpsi'};
fprintf('Trajectory from %s: N = %d, t_f = %.2f s\n\n', S.created, S.N, S.tf);
fprintf('Max |sim - optimizer| over the nodes (collocation error):\n');
for i = 1:10, fprintf('  %-5s %.2e\n', names{i}, max(err(i, :))); end

xf_err = X1(end, :).' - S.x(:, end);
fprintf('\nEnd of maneuver, sim - target: x %+.2f mm, l %+.2f mm, th %+.3f deg, phi %+.3f deg, psi %+.3f deg\n', ...
        1000*xf_err(1), 1000*xf_err(2), rad2deg(xf_err(3)), rad2deg(xf_err(4)), rad2deg(xf_err(5)));

coast = t > S.tf;
Q  = payload_xyz(X);
Qf = Q(:, find(~coast, 1, 'last'));
fprintf('Residual swing during the coast: max payload drift %.2f mm, max |phi| %.3f deg, max |psi| %.3f deg\n', ...
        1000*max(vecnorm(Q(1:2, coast) - Qf(1:2))), rad2deg(max(abs(X(4, coast)))), rad2deg(max(abs(X(5, coast)))));

viol = {};
for i = 1:10
    if any(X(i, ~coast) < S.x_lb(i) - 1e-6 | X(i, ~coast) > S.x_ub(i) + 1e-6), viol{end+1} = names{i}; end %#ok<SAGROW>
end
if any(any(U(:, ~coast) < S.u_lb - 1e-9 | U(:, ~coast) > S.u_ub + 1e-9)), viol{end+1} = 'u'; end
if isempty(viol), fprintf('Limits: all respected\n'); else, fprintf('Limits VIOLATED: %s\n', strjoin(viol, ', ')); end

%% 3. Plots (solid = sim, dots = optimizer nodes)
figure('Name', 'Crane trajectory', 'Color', 'w', 'Position', [60 60 1150 750]);
if isfield(S, 'w_t'), wt_str = sprintf('   (free t_f, w_t = %g)', S.w_t); else, wt_str = ''; end
sgtitle(sprintf('Optimal t_f = %.3f s%s', S.tf, wt_str), 'FontWeight', 'bold');
sub = @(k, rows, scale, lbl, ttl) plot_states(k, t, X(rows, :)*scale, S.t, S.x(rows, :)*scale, lbl, ttl, S.tf);
sub(1, 1:2, 1000,        'mm',  'Trolley x, cable l');
sub(2, 3,   180/pi,      'deg', 'Slew \theta');
sub(3, 4:5, 180/pi,      'deg', 'Swing \phi, \psi');
sub(4, 6:7, 1000,        'mm/s', 'Rates dx, dl');

subplot(2, 3, 5); hold on; grid on;
plot(t, U(1, :)*1000, t, U(2, :)*1000, t, rad2deg(U(3, :)), 'LineWidth', 1.5);
xline(S.tf, 'k-.', sprintf('t_f = %.2f s', S.tf));  xlabel('t (s)');  ylabel('mm/s^2, deg/s^2');  title('Inputs u');
legend('ddx', 'ddl', 'dd\theta', 'Location', 'best');

subplot(2, 3, 6); hold on; grid on; axis equal;
P = [X(1, :).*cos(X(3, :)); X(1, :).*sin(X(3, :))];
plot(1000*P(1, :), 1000*P(2, :), 'Color', [.6 .6 .6], 'LineWidth', 1.2);
plot(1000*Q(1, :), 1000*Q(2, :), 'b', 'LineWidth', 1.5);
plot(1000*Q(1, 1), 1000*Q(2, 1), 'go', 'MarkerFaceColor', 'g');
plot(1000*Qf(1), 1000*Qf(2), 'ks', 'MarkerFaceColor', 'k');
xlabel('X (mm)');  ylabel('Y (mm)');  title('Top view');  legend('trolley', 'payload', 'start', 'end', 'Location', 'best');

%% 4. Animation
if animate
    tA = 0 : 1/fps : t(end);
    XA = interp1(t, X.', tA).';
    QA = payload_xyz(XA);  PA = [XA(1, :).*cos(XA(3, :)); XA(1, :).*sin(XA(3, :)); zeros(size(tA))];
    H  = max(S.x_ub(2), 1) + 0.2;                            % jib height above the floor (drawing only)
    lift = [0; 0; H];
    fa = figure('Name', 'Crane animation', 'Color', 'w');  ax = axes(fa);  hold(ax, 'on');  grid(ax, 'on');
    axis(ax, 'equal');  view(ax, 40, 25);
    R = 1.2*max(S.x_ub(1), 0.5);
    xlim(ax, [-R R]);  ylim(ax, [-R R]);  zlim(ax, [0 H + 0.1]);  xlabel(ax, 'X (m)');  ylabel(ax, 'Y (m)');  zlabel(ax, 'Z (m)');
    plot3(ax, [0 0], [0 0], [0 H], 'Color', [.3 .3 .3], 'LineWidth', 5);
    hj = plot3(ax, [0 0], [0 0], [H H], 'Color', [.85 .6 .1], 'LineWidth', 3);
    hc = plot3(ax, [0 0], [0 0], [0 0], 'k-', 'LineWidth', 1.5);
    hp = plot3(ax, 0, 0, 0, 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 9);
    ht = plot3(ax, 0, 0, 0, 'r-');
    for k = 1:numel(tA)
        if ~isvalid(fa), break; end
        th = XA(3, k);  Pk = PA(:, k) + lift;  Qk = QA(:, k) + lift;
        set(hj, 'XData', [0 S.x_ub(1)*cos(th)], 'YData', [0 S.x_ub(1)*sin(th)], 'ZData', [H H]);
        set(hc, 'XData', [Pk(1) Qk(1)], 'YData', [Pk(2) Qk(2)], 'ZData', [Pk(3) Qk(3)]);
        set(hp, 'XData', Qk(1), 'YData', Qk(2), 'ZData', Qk(3));
        set(ht, 'XData', QA(1, 1:k), 'YData', QA(2, 1:k), 'ZData', QA(3, 1:k) + H);
        if tA(k) > S.tf, phase = 'coast (u = 0)'; else, phase = 'maneuver'; end
        title(ax, sprintf('%s   t = %.2f s', phase, tA(k)));
        drawnow;  pause(1/fps);
    end
end

%% Local functions
function Q = payload_xyz(X)
% Payload position in the ground frame (z relative to the jib, negative down), 3 x N.
x = X(1, :);  l = X(2, :);  th = X(3, :);  phi = X(4, :);  psi = X(5, :);
rx = x + l.*sin(phi).*cos(psi);                 % in the jib frame
ry = l.*sin(psi);
Q  = [rx.*cos(th) - ry.*sin(th);  rx.*sin(th) + ry.*cos(th);  -l.*cos(phi).*cos(psi)];
end

function plot_states(k, t, Y, tn, Yn, lbl, ttl, tf)
subplot(2, 3, k);  hold on;  grid on;
plot(t, Y, 'LineWidth', 1.5);
set(gca, 'ColorOrderIndex', 1);
plot(tn, Yn, '.', 'MarkerSize', 10);
xline(tf, 'k-.', sprintf('t_f = %.2f s', tf));  xlabel('t (s)');  ylabel(lbl);  title(ttl);
end
