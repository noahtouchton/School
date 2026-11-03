%% ME 6404 Lab 4 - Report figures (Parts 3-6)
% Runs the SAME plant (dequations.m) and the SAME control laws as source.m,
% but offline: no animation, no keyboard, so every run is repeatable and
% every state can be logged and plotted.
%
% How the simulation mimics source.m / PENDULUM.m:
%   * The controller is evaluated once every dt seconds (dt = min_cycle) and
%     the input is held constant in between (zero-order hold).
%   * Between updates ode45 integrates dequations.m, exactly like
%     PENDULUM.UPDATE_OTHER_DATA does.
%   * In 'Velocity' mode the commanded velocity is differentiated,
%     a = (U - U_old)/dt, and fed to the acceleration equations, just like
%     PENDULUM.m does.
%   * Disturbances are added to the input the same way the arrow keys are
%     (U = U_manual + U_fbl + U_lqr).
%
% Run section by section (Ctrl+Enter) to follow along, or run it all.
% Figures are saved as PNGs in ./figures.

clear; clc; close all;

%% 0. Parameters (same as source.m)
P.L  = 2.009;      % pendulum length [m]
P.mc = 3;          % cart mass [kg]
P.mp = 0.4781;     % payload mass [kg]
P.b  = 0.0513;     % damping
P.g  = 9.81;       % gravity [m/s^2]

dt   = 0.05;       % control update period = min_cycle in source.m [s]

saveFigs = true;
outDir   = fullfile(fileparts(mfilename('fullpath')), 'figures');
if saveFigs && ~exist(outDir, 'dir'), mkdir(outDir); end

noDist = @(t) 0;   % "no keys pressed"

%% 1. Gains
% Feedback-linearization gains (source.m: k0 = 16, k1 = 8).
% Closed loop theta dynamics become  theta_dd + k1*theta_d + k0*theta = 0
k0 = 16;
k1 = 8;

% --- Force-domain LQR (Part 4): linearized force model about theta = 0 ---
% Linearize dequations.m 'Force' at q = 0:
%   theta_dd = g(mc+mp)/(L mc)*theta - b/(mp L)*theta_d + F/(L mc)
%   x_dd     = mp g/mc*theta + F/mc
A_F = [0                         1               0 0;
       P.g*(P.mc+P.mp)/(P.L*P.mc) -P.b/(P.mp*P.L) 0 0;
       0                         0               0 1;
       P.mp*P.g/P.mc             0               0 0];
B_F = [0; 1/(P.L*P.mc); 0; 1/P.mc];

% Bryson's rule, same max values as klqr.m
Q_F = diag(1./[5*pi/180, 3.14, 1, 100].^2);   % [theta w x v] max values
R_F = 1/1^2;                                   % u_max = 1
K_F = lqr(A_F, B_F, Q_F, R_F);

% --- Velocity-domain LQR (Part 6): states z = [int(theta); theta; x] ---
% Integrate theta_dd = g/L theta - b/(mp L) theta_d + a/L once, with a = v_dot:
%   theta_d = g/L int(theta) - b/(mp L) theta + v/L,   x_d = v
A_V = [0      1               0;
       P.g/P.L -P.b/(P.mp*P.L) 0;
       0      0               0];
B_V = [0; 1/P.L; 1];

% Bryson's rule. theta_max = 10 deg, x_max = 2 m reproduce the source.m gains
% (~[94.8 71.3 -5]). NOTE: velLQR.m as saved (theta_max = 5 deg, x_max = 10 m)
% puts a pole near -57 rad/s, which is too fast for a 0.05 s update and the
% sampled loop goes unstable. Keep the fastest pole well below ~1/dt.
Q_V = diag(1./[1, 10*pi/180, 2].^2);           % [int(theta) theta x] max values
R_V = 1/10^2;                                  % u_max = 10 m/s
K_V = lqr(A_V, B_V, Q_V, R_V);

fprintf('Force LQR    K_F = [%s]\n', num2str(K_F, '%10.4f'));
fprintf('Velocity LQR K_V = [%s]\n', num2str(K_V, '%10.4f'));
fprintf('Force LQR closed-loop poles:    %s\n', num2str(eig(A_F - B_F*K_F).', '%8.3f'));
fprintf('Velocity LQR closed-loop poles: %s\n', num2str(eig(A_V - B_V*K_V).', '%8.3f'));
sdV = c2d(ss(A_V, B_V, eye(3), 0), dt);          % sampled-loop sanity check
fprintf('Velocity LQR max |z| at dt = %.2f s: %.3f (must be < 1)\n\n', ...
        dt, max(abs(eig(sdV.A - sdV.B*K_V))));

%% ===================== PART 3: FEEDBACK LINEARIZATION =====================
% Acceleration input a = x_dd. Theta equation (dequations.m):
%   theta_dd = g/L sin(th) - b/(mp L) w + cos(th)/L * a
% Choose a so the nonlinear terms cancel and theta_dd = -k0*th - k1*w:
%   a = L/cos(th) * ( -k0 th - k1 w - g/L sin(th) + b/(mp L) w )
% Note the 1/cos(th): the law is singular at th = +-90 deg (Part 3 Q7).

%% Fig 1 - Eigenvalues in the s-plane
% Theta subsystem in closed loop: [th; w]_dot = [0 1; -k0 -k1][th; w]
A_th = [0 1; -k0 -k1];
lam_th = eig(A_th);

% Full 4-state linearization of the closed loop: the cart only sees x_dd = a,
% so its two poles sit at the origin (double integrator, nothing controls x).
c  = P.b/(P.mp*P.L);
A_full = [0                  1             0 0;
          -k0                -k1           0 0;
          0                  0             0 1;
          -P.L*(k0+P.g/P.L)  -P.L*(k1-c)   0 0];
lam_full = eig(A_full);
lam_cart = lam_full(abs(lam_full) < 1e-6);

wn   = sqrt(k0);
zeta = k1/(2*sqrt(k0));
fprintf('Part 3 FBL: wn = %.2f rad/s, zeta = %.2f, poles = %s\n', ...
        wn, zeta, num2str(lam_th.', '%8.3f'));

f1 = figure('Name', 'Fig1_FBL_eigenvalues', 'Color', 'w');
hold on; grid on; box on;
plot(real(lam_th), imag(lam_th), 'x', 'MarkerSize', 14, 'LineWidth', 2.5);
plot(real(lam_cart), imag(lam_cart), 'o', 'MarkerSize', 10, 'LineWidth', 2);
xline(0, 'k-', 'HandleVisibility', 'off'); yline(0, 'k-', 'HandleVisibility', 'off');
% constant-wn circle for reference
phi = linspace(pi/2, 3*pi/2, 200);
plot(wn*cos(phi), wn*sin(phi), ':', 'Color', [0.5 0.5 0.5], 'HandleVisibility', 'off');
axis equal
xlim([-1.5*wn, 0.5*wn]); ylim([-1.2*wn, 1.2*wn]);
xlabel('Real [1/s]'); ylabel('Imaginary [rad/s]');
title('Closed-Loop Eigenvalues');
legend({'\theta, \theta''', 'x, x'''}, 'Location', 'northwest');
saveFig(f1, outDir, saveFigs);

%% Fig 2 - Righting from hanging + cart drift (acceleration mode)
th0 = pi - 0.05;                         % hanging down (slightly off pi)
ctrlFBLa = @(q, t, mem) fblAccel(q, mem, P, k0, k1);
S3 = simulate(P, 'Acceleration', [th0 0 0 0], 6, dt, ctrlFBLa, noDist);

% Linear prediction from the eigenvalues, to compare with the nonlinear sim
sysTh   = ss(A_th, zeros(2,1), eye(2), 0);
qPred   = initial(sysTh, [th0; 0], S3.t);

f2 = plotStates(S3, 'Feedback Linearization Righting', ...
                'a [m/s^2]', 'Fig2_FBL_righting_cart_drift');
nexttile(1); hold on; plot(S3.t, rad2deg(qPred(:,1)), 'k--');
legend('Simulation', 'Linear prediction');
nexttile(2); hold on; plot(S3.t, rad2deg(qPred(:,2)), 'k--');
saveFig(f2, outDir, saveFigs);

%% ========================= PART 4: FORCE LQR ==============================
% u = F = -K_F * [th; w; x; v].  Start near upright.

%% Fig 3 - Upright regulation from a small tilt
ctrlLQRf = @(q, t, mem) lqrForce(q, mem, K_F);
S4 = simulate(P, 'Force', [deg2rad(5) 0 0.5 0], 10, dt, ctrlLQRf, noDist);
f3 = plotStates(S4, 'Force LQR Regulation', ...
                'F [N]', 'Fig3_LQR_force_regulation');
saveFig(f3, outDir, saveFigs);

%% Fig 4 - External force disturbance rejection
Fd     = 20;            % disturbance force [N]
tPulse = [2 2.2];       % pulse window [s]
distF  = @(t) Fd * (t >= tPulse(1) && t < tPulse(2));
S4d = simulate(P, 'Force', [0 0 0 0], 10, dt, ctrlLQRf, distF);
f4 = plotStates(S4d, 'Force LQR Disturbance Rejection', ...
                'F [N]', 'Fig4_LQR_force_disturbance');
shadeWindow(f4, tPulse);
saveFig(f4, outDir, saveFigs);

% Actuator check (Control Input History): peak force for both runs
fprintf('Part 4 peak |F|: regulation = %.1f N, disturbance = %.1f N\n', ...
        max(abs(S4.U)), max(abs(S4d.U)));

%% =================== PART 5: COMBINED FBL + LQR (FORCE) ===================
% Force-domain FBL: solve the force theta equation for F so that
% theta_dd = v = -k0 th - k1 w:
%   D = mc + mp sin^2(th)
%   F = L D/cos(th) * (v - g/L sin(th) + b/(mp L) w) + mp L w^2 sin(th) - mp g sin(th) cos(th)
% Switch to LQR once |th| < thCatch; fall back to FBL if |th| > thDrop.
% Softer FBL gains than Part 3 (wn = 3, zeta = 1): with k0 = 16, k1 = 8 the
% pendulum arrives too fast, LQR can't hold it and the modes chatter.
k0f = 9;
k1f = 6;
thCatch = deg2rad(15);
thDrop  = deg2rad(60);
ctrlComb = @(q, t, mem) combined(q, mem, P, k0f, k1f, K_F, thCatch, thDrop);

%% Fig 5 - Swing-up and catch from hanging
S5 = simulate(P, 'Force', [pi-0.05 0 0 0], 20, dt, ctrlComb, noDist);
f5 = plotStates(S5, 'Combined Swing-Up and Catch', ...
                'F [N]', 'Fig5_combined_swingup_catch');
shadeMode(f5, S5);
saveFig(f5, outDir, saveFigs);
tSw = S5.t(find(S5.active == 2, 1));
fprintf('Part 5: LQR takes over at t = %.2f s, peak |x| = %.2f m, peak |F| = %.1f N\n', ...
        tSw, max(abs(S5.Q(:,3))), max(abs(S5.U)));

%% Fig 6 - Robustness sweep: initial angles and disturbance sizes
% Avoid starting right at +-90 deg: cos(th) = 0 is the FBL singularity and the
% force command blows up (try 90.5 to see it; that's your Part 3 Q7 answer).
th0List = deg2rad([-150 -60 15 45 100 135 177]);
FdList  = [10 20 40 60 80];                  % 0.2 s force pulses at t = 2 s, start upright

f6 = figure('Name', 'Fig6_combined_robustness', 'Color', 'w', 'Position', [100 100 1100 650]);
tl = tiledlayout(2, 2, 'TileSpacing', 'compact');
title(tl, 'Combined Control Robustness');
cmap = lines(max(numel(th0List), numel(FdList)));

ax = gobjects(4,1);
for i = 1:4, ax(i) = nexttile(i); hold on; grid on; box on; end
for i = 1:numel(th0List)
    S = simulate(P, 'Force', [th0List(i) 0 0 0], 20, dt, ctrlComb, noDist);
    lbl = sprintf('%.0f^\\circ', rad2deg(th0List(i)));
    plot(ax(1), S.t, rad2deg(wrapPi(S.Q(:,1))), 'Color', cmap(i,:), 'DisplayName', lbl);
    plot(ax(3), S.t, S.Q(:,3),                  'Color', cmap(i,:), 'DisplayName', lbl);
end
for i = 1:numel(FdList)
    d = @(t) FdList(i) * (t >= 2 && t < 2.2);
    S = simulate(P, 'Force', [0 0 0 0], 20, dt, ctrlComb, d);
    lbl = sprintf('%g N', FdList(i));
    plot(ax(2), S.t, rad2deg(wrapPi(S.Q(:,1))), 'Color', cmap(i,:), 'DisplayName', lbl);
    plot(ax(4), S.t, S.Q(:,3),                  'Color', cmap(i,:), 'DisplayName', lbl);
end
ylabel(ax(1), '\theta [deg]');
ylabel(ax(2), '\theta [deg]');
ylabel(ax(3), 'x [m]'); xlabel(ax(3), 'Time [s]');
ylabel(ax(4), 'x [m]'); xlabel(ax(4), 'Time [s]');
legend(ax(1), 'Location', 'northeast'); legend(ax(2), 'Location', 'northeast');
saveFig(f6, outDir, saveFigs);

%% ======================== PART 6: VELOCITY LQR ============================
% u = v_cmd = -K_V * [int(th); th; x], with the same leaky integrator as
% source.m (int = 0.99*int + th*dt) to stop numerical wind-up.

%% Fig 7 - Upright regulation (velocity input)
ctrlLQRv = @(q, t, mem) lqrVelocity(q, mem, K_V);
S6 = simulate(P, 'Velocity', [deg2rad(5) 0 0.5 0], 10, dt, ctrlLQRv, noDist);
f7 = plotVelocity(S6, 'Velocity LQR Regulation', ...
                  'Fig7_LQR_velocity_regulation');
saveFig(f7, outDir, saveFigs);

%% Fig 8 - Velocity disturbance rejection
Vd     = 1;             % same size as one arrow-key press in velocity mode [m/s]
tPulse = [2 2.5];
distV  = @(t) Vd * (t >= tPulse(1) && t < tPulse(2));
S6d = simulate(P, 'Velocity', [0 0 0 0], 10, dt, ctrlLQRv, distV);
f8 = plotVelocity(S6d, 'Velocity LQR Disturbance Rejection', ...
                  'Fig8_LQR_velocity_disturbance');
shadeWindow(f8, tPulse);
saveFig(f8, outDir, saveFigs);
fprintf('Part 6 peak |v_cmd|: regulation = %.2f m/s, disturbance = %.2f m/s\n', ...
        max(abs(S6.U)), max(abs(S6d.U)));

if saveFigs, fprintf('\nFigures saved to %s\n', outDir); end


%% ============================ LOCAL FUNCTIONS =============================

function S = simulate(P, Mode, q0, Tend, dt, ctrl, dist)
% Zero-order-hold simulation of dequations.m, mirroring PENDULUM.m.
%   ctrl : @(q, t, mem) -> [u, mem]   controller (mem carries integrator etc.)
%   dist : @(t) -> extra input added to u (like the arrow keys)
N  = round(Tend/dt);
t  = (0:N)' * dt;
Q  = zeros(N+1, 4);  Q(1,:) = q0(:)';
U  = zeros(N+1, 1);
active = zeros(N+1, 1);                 % 1 = FBL, 2 = LQR (for Part 5 shading)
mem = struct('int', 0, 'dt', dt, 'active', 0);
uOld = 0;
plantMode = Mode;
if strcmp(Mode, 'Velocity'), plantMode = 'Acceleration'; end

for k = 1:N
    [u, mem] = ctrl(Q(k,:)', t(k), mem);
    u = u + dist(t(k));
    U(k) = u;  active(k) = mem.active;

    if strcmp(Mode, 'Velocity')
        uPlant = (u - uOld)/dt;  uOld = u;   % PENDULUM.m differentiates v_cmd
    else
        uPlant = u;
    end

    [~, y] = ode45(@(tt, q) dequations(tt, q, uPlant, P.L, P.mp, P.mc, P.b, plantMode), ...
                   [0 dt], Q(k,:)');
    Q(k+1,:) = y(end,:);
end
U(end) = U(end-1);  active(end) = active(end-1);
S = struct('t', t, 'Q', Q, 'U', U, 'active', active, 'Mode', Mode);
end

function [u, mem] = fblAccel(q, mem, P, k0, k1)
% Part 3: feedback linearization, acceleration input (same law as source.m)
th = wrapPi(q(1));  w = q(2);
u = P.L/cos(th) * (-k0*th - k1*w - P.g*sin(th)/P.L + P.b*w/(P.mp*P.L));
mem.active = 1;
end

function [u, mem] = lqrForce(q, mem, K)
% Part 4: full-state LQR, force input
q(1) = wrapPi(q(1));
u = -K * q;
mem.active = 2;
end

function F = fblForce(q, P, k0, k1)
% Part 5: feedback linearization, force input
th = wrapPi(q(1));  w = q(2);
st = sin(th);  ct = cos(th);
D  = P.mc + P.mp*st^2;
v  = -k0*th - k1*w;                       % desired theta_dd
F  = P.L*D/ct * (v - P.g/P.L*st + P.b/(P.mp*P.L)*w) ...
     + P.mp*P.L*w^2*st - P.mp*P.g*st*ct;
end

function [u, mem] = combined(q, mem, P, k0, k1, K, thCatch, thDrop)
% Part 5: FBL far from upright, LQR near upright (with hysteresis)
th = abs(wrapPi(q(1)));
if mem.active == 2 && th > thDrop
    mem.active = 1;
elseif mem.active ~= 2 && th < thCatch
    mem.active = 2;
elseif mem.active == 0
    mem.active = 1;
end
if mem.active == 2
    q(1) = wrapPi(q(1));
    u = -K * q;
else
    u = fblForce(q, P, k0, k1);
end
end

function [u, mem] = lqrVelocity(q, mem, K)
% Part 6: velocity LQR on [int(theta); theta; x] with leaky integrator
th = wrapPi(q(1));
mem.int = mem.int*0.99 + th*mem.dt;
u = -K * [mem.int; th; q(3)];
mem.active = 2;
end

function a = wrapPi(a)
% wrap angle to (-pi, pi]
a = atan2(sin(a), cos(a));
end

function f = plotStates(S, ttl, uLabel, name)
% 5 stacked plots: theta, theta_dot, x, x_dot, input
f = figure('Name', name, 'Color', 'w', 'Position', [100 100 800 900]);
tl = tiledlayout(5, 1, 'TileSpacing', 'compact');
title(tl, ttl);
data = {rad2deg(wrapPi(S.Q(:,1))), rad2deg(S.Q(:,2)), S.Q(:,3), S.Q(:,4)};
lbls = {'\theta [deg]', '\theta'' [deg/s]', 'x [m]', 'x'' [m/s]'};
for i = 1:4
    nexttile; plot(S.t, data{i}, 'LineWidth', 1.4); grid on; ylabel(lbls{i});
end
nexttile; stairs(S.t, S.U, 'LineWidth', 1.4); grid on; ylabel(uLabel);
xlabel('Time [s]');
end

function f = plotVelocity(S, ttl, name)
% Velocity-domain plot: LQR states [int(theta), theta, x], plus x_dot and v_cmd
f = figure('Name', name, 'Color', 'w', 'Position', [100 100 800 900]);
tl = tiledlayout(5, 1, 'TileSpacing', 'compact');
title(tl, ttl);
th = wrapPi(S.Q(:,1));
intTh = zeros(size(th));                  % rebuild the controller's integrator
for k = 2:numel(th)
    intTh(k) = intTh(k-1)*0.99 + th(k-1)*(S.t(k)-S.t(k-1));
end
data = {rad2deg(intTh), rad2deg(th), S.Q(:,3), S.Q(:,4)};
lbls = {'\int\theta dt [deg s]', '\theta [deg]', 'x [m]', 'x'' [m/s]'};
for i = 1:4
    nexttile; plot(S.t, data{i}, 'LineWidth', 1.4); grid on; ylabel(lbls{i});
end
nexttile; stairs(S.t, S.U, 'LineWidth', 1.4); grid on; ylabel('v_{cmd} [m/s]');
xlabel('Time [s]');
end

function shadeWindow(f, tw)
% grey band over the disturbance window on every axis
for ax = findobj(f, 'Type', 'axes')'
    yl = ylim(ax);
    patch(ax, [tw(1) tw(2) tw(2) tw(1)], [yl(1) yl(1) yl(2) yl(2)], [0.6 0.6 0.6], ...
          'FaceAlpha', 0.25, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    ylim(ax, yl);
end
end

function shadeMode(f, S)
% mark the FBL -> LQR handoff on every axis
iSw = find(S.active == 2, 1);
if isempty(iSw), return; end
for ax = findobj(f, 'Type', 'axes')'
    yl = ylim(ax);
    patch(ax, [0 S.t(iSw) S.t(iSw) 0], [yl(1) yl(1) yl(2) yl(2)], [1 0.8 0.4], ...
          'FaceAlpha', 0.25, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    xline(ax, S.t(iSw), 'k--', 'HandleVisibility', 'off');
    ylim(ax, yl);
end
end

function saveFig(f, outDir, doSave)
if doSave
    exportgraphics(f, fullfile(outDir, [f.Name '.png']), 'Resolution', 200);
end
end
