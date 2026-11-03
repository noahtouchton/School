%=========================================================
%  LAB 3: MODEL REFERENCE CONTROL -- all figures
%
%  Model reference responses come from the provided Simulink model
%  (student_model_ref.mdl + student_model_ref_controller.m).
%  Crane data is read from Analysis/Data, figures go to Analysis/figures.
%
%  x2 = L*theta (m) is the payload deflection, u is trolley velocity (m/s).
%  The crane logs u and x2 but not the operator command v (that column is
%  all zeros), so v is rebuilt from u = v + k1*x1 + k2*x2 and snapped to
%  the +/-100% button speed. Each crane run is one move out and one move
%  back; the Simulink model makes one pulse per run, so the two pulses are
%  simulated separately and added (the system is linear, zero initial state).
%==========================================================
clear; clc; close all;

here = fileparts(mfilename('fullpath'));
addpath(here);
dataDir = fullfile(here, 'Analysis', 'Data');
outDir  = fullfile(here, 'Analysis', 'figures');
if ~exist(outDir, 'dir'), mkdir(outDir); end

% run list: file suffix, zeta_m, wm/wn, actual cable length L
part1 = [mk('0.1', 0.1, 1, 1), mk('0.5', 0.5, 1, 1), mk('0.8', 0.8, 1, 1)];

zetaSweep = [];
for z = [0.6 0.7 0.9 1.0 1.1 1.2 1.5]
    zetaSweep = [zetaSweep, mk(sprintf('i_%.1f', z), z, 1, 1)]; %#ok<AGROW>
end

wmSweep = [];
for p = [5 10 25 35 50]
    wmSweep = [wmSweep, mk(sprintf('ii_%d%%', p), 0.5, 1 + p/100, 1)]; %#ok<AGROW>
end

lenSweep01 = [];
for Lr = [0.6 0.8 1.0 1.2 1.4]
    lenSweep01 = [lenSweep01, mk(lenFile('0.1', Lr), 0.1, 1, Lr)]; %#ok<AGROW>
end
lenSweep05 = [];
for Lr = [0.8 1.0 1.2 1.4]
    lenSweep05 = [lenSweep05, mk(lenFile('0.5', Lr), 0.5, 1, Lr)]; %#ok<AGROW>
end

zlab = @(r) sprintf('\\zeta_m = %.1f', r.zm);
wlab = @(r) sprintf('\\omega_m = %.2f\\omega_n', r.ratio);
llab = @(r) sprintf('L = %.1f m', r.L);

%% 0) Simulation only (same pulse as lab3model.m)
simulation_figure(outDir);

%% 1) Part 1: zeta_m = 0.1, 0.5, 0.8 at L = 1 m
R1 = analyzeAll(part1, dataDir);
run_grid(R1, 'Part 1: experiment vs model reference, L = 1 m', zlab, fullfile(outDir, '1_part1_runs.png'));
overlay(R1, 'Part 1: measured deflection for each \zeta_m', zlab, fullfile(outDir, '1_part1_overlay.png'));

%% 2) Part 2(i): zeta_m sweep at wm = wn
R2 = analyzeAll(zetaSweep, dataDir);
run_grid(R2, 'Part 2(i): \zeta_m sweep, \omega_m = \omega_n', zlab, fullfile(outDir, '2_zeta_sweep_runs.png'));
R2all = [R1, R2];
[~, order] = sort([R2all.zm]);
R2all = R2all(order);
overlay(R2all, 'Part 2(i): measured deflection vs \zeta_m', zlab, fullfile(outDir, '2_zeta_sweep_overlay.png'));
metrics(R2all, [R2all.zm], '\zeta_m', 'Part 2(i): effect of \zeta_m (\omega_m = \omega_n)', ...
        fullfile(outDir, '2_zeta_sweep_metrics.png'));

%% 3) Part 2(ii): wm sweep at zeta_m = 0.5
R3 = analyzeAll(wmSweep, dataDir);
run_grid(R3, 'Part 2(ii): \omega_m sweep, \zeta_m = 0.5', wlab, fullfile(outDir, '3_wm_sweep_runs.png'));
R3all = [R1(2), R3];
overlay(R3all, 'Part 2(ii): measured deflection vs \omega_m (\zeta_m = 0.5)', wlab, ...
        fullfile(outDir, '3_wm_sweep_overlay.png'));
metrics(R3all, 100*([R3all.ratio] - 1), '\omega_m above \omega_n (%)', ...
        'Part 2(ii): effect of \omega_m (\zeta_m = 0.5)', fullfile(outDir, '3_wm_sweep_metrics.png'));

%% 4) Length robustness: controller designed for L = 1 m
sweeps = {lenSweep01, lenSweep05};
for k = 1:2
    R4 = analyzeAll(sweeps{k}, dataDir);
    zm = R4(1).zm;
    tag = strrep(sprintf('zeta%.1f', zm), '.', 'p');
    run_grid(R4, sprintf('Length robustness: controller designed for L = 1 m, \\zeta_m = %.1f', zm), ...
             llab, fullfile(outDir, ['4_length_' tag '_runs.png']));
    overlay(R4, sprintf('Length robustness: measured deflection, \\zeta_m = %.1f', zm), ...
            llab, fullfile(outDir, ['4_length_' tag '_overlay.png']));
    metrics(R4, [R4.L], 'actual cable length L (m)', ...
            sprintf('Length robustness (\\zeta_m = %.1f, designed for L = 1 m)', zm), ...
            fullfile(outDir, ['4_length_' tag '_metrics.png']));
end

fprintf('wrote %d figures to %s\n', numel(dir(fullfile(outDir, '*.png'))), outDir);


%% ======================= local functions =======================

function c = consts()
    c.g = 9.81;
    c.Ldesign = 1.0;      % controller gains were computed for L = 1 m
    c.zp = 0.001;         % plant damping
    c.vmax = 0.2;         % 100% trolley speed, m/s
    c.dropout = 600;      % camera pins at ~612 mm when it loses the payload
    c.unstable = 150;     % any clean deflection beyond this is a runaway
end

function r = mk(suffix, zm, ratio, L)
    r = struct('suffix', suffix, 'zm', zm, 'ratio', ratio, 'L', L);
end

function f = lenFile(base, L)
    if L == 1.0
        f = base;
    else
        f = sprintf('%s_L=%.1f', base, L);
    end
end

function R = analyzeAll(runs, dataDir)
    R = [];
    for r = runs
        fprintf('  %s\n', r.suffix);
        R = [R, analyze(r, dataDir)]; %#ok<AGROW>
    end
end

function r = analyze(r, dataDir)
    c = consts();
    [t, defl, vel, dropped] = load_run(fullfile(dataDir, ['GT_BridgeCrane_Data_' r.suffix '.csv']));
    wm = r.ratio * sqrt(c.g / c.Ldesign);
    v = rebuild_command(t, defl, vel, r.zm, wm);
    xd2 = model_reference(t, v, r.zm, wm) * 1000;

    ok = ~isnan(defl);
    r.t = t; r.defl = defl; r.vel = vel; r.v = v * 1000; r.xd2 = xd2;
    r.peak = max(abs(defl(ok)));
    r.peak_ref = max(abs(xd2));
    r.rms_err = sqrt(mean((defl(ok) - xd2(ok)).^2));
    r.unstable = dropped || r.peak > c.unstable;
end

function [t, defl, vel, dropped] = load_run(file)
    % Y-axis log, trimmed of the zero padding and duplicate timestamps
    c = consts();
    M = readmatrix(file);
    n = find(M(:, 7) ~= 0, 1, 'last');
    M = M(1:n, :);
    [t, ia] = unique(M(:, 7));
    defl = M(ia, 8);
    vel  = M(ia, 10);

    drop = abs(defl) >= c.dropout;
    dropped = any(drop);
    defl(drop) = NaN;

    % remove the camera's resting offset (mean before the first move)
    i0 = find(abs(vel) > 20, 1);
    if isempty(i0), t0 = t(end); else, t0 = t(i0); end
    rest = t < max(t0 - 0.1, t(2));
    defl = defl - mean(defl(rest), 'omitnan');
end

function v = rebuild_command(t, defl, vel, zm, wm)
    % recover the operator's pulse command v from u = v + k1 x1 + k2 x2
    c = consts();
    wn = sqrt(c.g / c.Ldesign);
    k1 = -(wm^2 - wn^2);
    k2 = -2*(zm*wm - c.zp*wn);

    x2 = defl / 1000;
    x2(isnan(x2)) = 0;
    u  = vel / 1000;
    x1 = cumtrapz(t, x2);
    vraw = u - k1*x1 - k2*x2;
    v = c.vmax*(vraw > c.vmax/2) - c.vmax*(vraw < -c.vmax/2);

    % bridge short gaps inside one button press (feedback dips u below v/2)
    S = segments(v);
    for i = 2:size(S, 1) - 1
        a = S(i, 1); b = S(i, 2);
        prev = v(S(i-1, 1)); next = v(S(i+1, 1));
        if v(a) == 0 && prev ~= 0 && prev == next && t(b) - t(a) < 0.5
            v(a:b) = prev;
        end
    end

    % one move out and one move back: keep the first press in each
    % direction and drop the rest (blips, or feedback once a run goes unstable)
    seen = [];
    S = segments(v);
    for i = 1:size(S, 1)
        a = S(i, 1); b = S(i, 2); s = v(a);
        if s == 0, continue; end
        if any(seen == s) || t(b) - t(a) < 0.3
            v(a:b) = 0;
        else
            seen(end+1) = s; %#ok<AGROW>
        end
    end
end

function S = segments(v)
    % [first last] index of each run of constant value
    edges = find(diff(v(:)) ~= 0);
    S = [[1; edges + 1], [edges; numel(v)]];
end

function xd2 = model_reference(t, v, zm, wm)
    % drive student_model_ref with each command pulse and add the results
    xd2 = zeros(size(t));
    S = segments(v);
    for i = 1:size(S, 1)
        a = S(i, 1); b = S(i, 2);
        if v(a) == 0, continue; end
        tOff = t(min(b + 1, numel(t)));   % v(k) holds from t(k) to t(k+1)
        out = run_model(zm, wm, v(a), t(a), tOff, t(end));
        xd2 = xd2 + interp1(out.t, out.xd(:, 2), t, 'linear', 0);
    end
end

function out = run_model(zm, wm, vStep, tOn, tOff, tSim)
    % one run of the provided Simulink model: v = vStep from tOn to tOff
    global Ap A P %#ok<GVMIS> used by student_model_ref_controller
    c = consts();
    wn = sqrt(c.g / c.Ldesign);

    Ap = [0 1; -wn^2 -2*c.zp*wn];  Bp = [0; 1];  Cp = eye(2);  Dp = [0; 0]; %#ok<NASGU>
    A  = [0 1; -wm^2 -2*zm*wm];    B  = [0; 1];  C  = eye(2);  D  = [0; 0]; %#ok<NASGU>
    Q = eye(2);
    P = reshape(-(kron(eye(2), A') + kron(A', eye(2))) \ Q(:), 2, 2);  % A'P + PA = -Q

    step1 = vStep;  step2 = -vStep;  t_on = tOn;  t_off = tOff;  t_sim = tSim; %#ok<NASGU>
    so = sim('student_model_ref', 'SrcWorkspace', 'current', ...
             'ReturnWorkspaceOutputs', 'on', 'MaxStep', '0.01');

    out.t  = so.xd.Time;
    out.xd = states(so.xd.Data);
    out.x  = states(so.x);
    out.u  = squeeze(so.u.Data);
end

function X = states(d)
    % Simulink may save 2-state signals as Nx2 or 2x1xN
    if isa(d, 'timeseries'), d = d.Data; end
    X = squeeze(double(d));
    if size(X, 2) ~= 2 && size(X, 1) == 2
        X = X';
    end
end


%% ------------------------------ plots ------------------------------

function simulation_figure(outDir)
    % pure simulation, same pulse as lab3model.m (0.2 m/s from 1 s to 5 s)
    % styled to match the experimental Model A/B/C deflection figure
    c = consts();
    wn = sqrt(c.g / c.Ldesign);
    fig = figure('Visible', 'off', 'Color', 'w', 'Position', [0 0 1100 1000]);
    tl = tiledlayout(fig, 2, 1, 'TileSpacing', 'loose');
    ax1 = nexttile(tl); hold(ax1, 'on');
    ax2 = nexttile(tl); hold(ax2, 'on');
    gray = [0.3 0.3 0.3];

    % A = Ap makes the controller gains zero, i.e. no control (u = v)
    open = run_model(c.zp, wn, c.vmax, 1, 5, 10);
    plot(ax1, open.t, open.x(:, 2)*1000, 'Color', gray, 'LineWidth', 1, ...
         'DisplayName', 'No control');
    plot(ax2, open.t, open.u*1000, 'Color', gray, 'LineWidth', 1, 'DisplayName', 'Command v');

    cols = lines(3);                      % same default colors as Model A/B/C
    names = {'Model A', 'Model B', 'Model C'};
    zms = [0.1 0.5 0.8];
    for i = 1:3
        o = run_model(zms(i), wn, c.vmax, 1, 5, 10);
        plot(ax1, o.t, o.xd(:, 2)*1000, 'Color', cols(i, :), 'LineWidth', 1.5, ...
             'DisplayName', names{i});
        plot(ax2, o.t, o.u*1000, 'Color', cols(i, :), 'LineWidth', 1.5, ...
             'DisplayName', names{i});
    end

    ylabel(ax1, 'Y-direction deflection [mm]');
    ylabel(ax2, 'Y-direction velocity [mm/s]');
    ylim(ax1, [-80 80]); yticks(ax1, -80:20:80);
    ylim(ax2, [-50 250]); yticks(ax2, -50:50:250);
    for ax = [ax1 ax2]
        xlabel(ax, 'Time [sec]');
        xlim(ax, [0 10]); xticks(ax, 0:2:10);
        grid(ax, 'on');
        set(ax, 'FontSize', 11);
        legend(ax, 'Location', 'northeast', 'FontSize', 10);
    end
    exportgraphics(fig, fullfile(outDir, '0_simulation_zeta.png'), 'Resolution', 200);
    close(fig);
end

function run_grid(R, ttl, labelFn, file)
    % one row per run: deflection vs reference | trolley velocity vs command
    n = numel(R);
    fig = figure('Visible', 'off', 'Position', [0 0 1200 230*n + 80]);
    tl = tiledlayout(fig, n, 2, 'TileSpacing', 'compact');
    blue = [0 0.447 0.741];
    for k = 1:n
        r = R(k);
        axd = nexttile(tl); hold(axd, 'on');
        plot(axd, r.t, r.defl, 'Color', blue, 'LineWidth', 1.4, 'DisplayName', 'measured');
        plot(axd, r.t, r.xd2, 'k--', 'LineWidth', 1.3, 'DisplayName', 'model reference x_{d2}');
        tag = labelFn(r);
        if r.unstable
            title(axd, [tag '  (UNSTABLE)'], 'Color', [0.85 0.1 0.1]);
            ylim(axd, [-250 250]);
        else
            title(axd, tag);
        end
        ylabel(axd, 'deflection (mm)'); grid(axd, 'on');

        axv = nexttile(tl); hold(axv, 'on');
        plot(axv, r.t, r.vel, 'Color', blue, 'LineWidth', 1.4, 'DisplayName', 'trolley velocity u');
        stairs(axv, r.t, r.v, 'k--', 'LineWidth', 1.3, 'DisplayName', 'command v (rebuilt)');
        ylabel(axv, 'velocity (mm/s)'); grid(axv, 'on');

        if k == 1
            legend(axd, 'Location', 'northeast', 'FontSize', 8);
            legend(axv, 'Location', 'northeast', 'FontSize', 8);
        end
        if k == n
            xlabel(axd, 'time (s)'); xlabel(axv, 'time (s)');
        end
    end
    title(tl, ttl);
    exportgraphics(fig, file, 'Resolution', 200);
    close(fig);
end

function overlay(R, ttl, labelFn, file)
    % all runs of a sweep on one deflection axis (unstable runs clipped)
    fig = figure('Visible', 'off', 'Position', [0 0 1000 500]);
    ax = axes(fig); hold(ax, 'on');
    cols = parula(numel(R) + 1);
    for k = 1:numel(R)
        r = R(k);
        name = labelFn(r);
        ls = '-';
        if r.unstable, name = [name ' (unstable)']; ls = ':'; end
        plot(ax, r.t, r.defl, ls, 'Color', cols(k, :), 'LineWidth', 1.4, 'DisplayName', name);
    end
    ylim(ax, [-150 150]);
    xlabel(ax, 'time (s)'); ylabel(ax, 'payload deflection (mm)');
    title(ax, ttl); grid(ax, 'on'); legend(ax, 'Location', 'northeast');
    exportgraphics(fig, file, 'Resolution', 200);
    close(fig);
end

function metrics(R, x, xlab, ttl, file)
    bad = [R.unstable];
    fig = figure('Visible', 'off', 'Position', [0 0 1100 400]);
    tl = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact');
    keys = {'peak', 'rms_err'};
    ylabs = {'peak |deflection| (mm)', 'RMS error vs reference (mm)'};
    for k = 1:2
        ax = nexttile(tl); hold(ax, 'on');
        y = [R.(keys{k})];
        plot(ax, x(~bad), y(~bad), 'o-', 'LineWidth', 1.4, 'DisplayName', 'stable');
        if any(bad)
            plot(ax, x(bad), min(y(bad), 250), 'x', 'Color', [0.85 0.1 0.1], ...
                 'MarkerSize', 12, 'LineWidth', 2, 'DisplayName', 'unstable (clipped)');
        end
        if k == 1
            plot(ax, x, [R.peak_ref], 'ks--', 'DisplayName', 'model reference');
        end
        xlabel(ax, xlab); ylabel(ax, ylabs{k}); grid(ax, 'on');
        legend(ax, 'Location', 'best');
    end
    title(tl, ttl);
    exportgraphics(fig, file, 'Resolution', 200);
    close(fig);
end
