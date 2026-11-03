clc; clear;

% =========================================================================
% --- 0. TUNABLE LAB VARIABLES (Adjust Here for Physical Machine) ---
% =========================================================================
t_wait       = 1.0;   % Waiting period (s) between each phase (Hoist -> Slew -> Hoist)
d_slew       = 40.0;  % Target slew angle traveled (deg) - increase (e.g. 42, 45) if undershooting
d_hoist_up   = 0.550; % Hoist UP distance (m)   [1350 mm -> 800 mm = 0.550 m]
d_hoist_down = 0.550; % Hoist DOWN distance (m) [800 mm -> 1350 mm = 0.550 m]

% =========================================================================
% --- 1. System Parameters & Limits (Table IV) ---
% =========================================================================
max_accel_hoist   = 30;   % mm/s^2
max_accel_slew    = 32.4; % deg/s^2
max_accel_trolley = 72.6; % mm/s^2

L  = 0.800;               % Rigging length (m)
g  = 9.81;                % Gravity (m/s^2)
wn = sqrt(g/L);

% Shaper Definition (Unity-Magnitude ZV / Negative Shaper)
amp_shaper = [1, -1, 1];
t_shaper   = [0, acos(0.5)/wn, acos(-0.5)/wn]; 

% =========================================================================
% --- 2. Dynamic Time Calculations (With Wait Periods Between Phases) ---
% =========================================================================

% --- PHASE 1: Hoist Up (Negative: -1) ---
v_hoist          = 0.03;                       % m/s (30 mm/s)
a_hoist          = max_accel_hoist * 1e-3;     % 0.03 m/s^2
t_hoist_accel    = v_hoist / a_hoist;          % t1 = 1.0 s

% Exact trapezoidal timing so total upward travel = d_hoist_up (0.550 m)
t_up_decel_start = d_hoist_up / v_hoist;               % 18.3333 s (t1 + t_cruise)
t_up_total       = t_up_decel_start + t_hoist_accel;   % 19.3333 s

% Switch times: [0.0; 1.0; 18.3333; 19.3333]
phase1_times = [0.0; t_hoist_accel; t_up_decel_start; t_up_total];
phase1_hoist_deltas = [
    -1;  % 0 -> -1 (Start hoist up: negative)
     1;  % -1 -> 0 (Reach cruise at t1 = 1.0 s)
     1;  % 0 -> +1 (Start decelerating hoist up)
    -1   % +1 -> 0 (Stop hoist completely at end of Phase 1)
];

% --- PHASE 2: Slew Maneuver (Starts after t_wait) ---
t_slew_start     = phase1_times(end) + t_wait; 
v_slew           = 32.4;                        % deg/s
a_slew           = max_accel_slew;              % 32.4 deg/s^2
t_slew_accel     = v_slew / a_slew;             % 1.0 s

% Trapezoidal slew timing using tunable d_slew
t_slew_decel_rel = d_slew / v_slew;                    % t_accel + t_cruise
t_slew_total_rel = t_slew_decel_rel + t_slew_accel;    % 2*t_accel + t_cruise

% Base unshaped slew switch times & step changes (deltas) relative to t_slew_start
slew_base_rel_times = [0; t_slew_accel; t_slew_decel_rel; t_slew_total_rel];
slew_base_deltas    = [-1; 1; 1; -1];           % Negative slew: 0->-1, -1->0, 0->+1, +1->0

% Apply Shaper Convolution to Slew Step Transitions via Matrix Broadcasting
slew_base_abs_times   = t_slew_start + slew_base_rel_times;
shaped_slew_times_mat = slew_base_abs_times + t_shaper;   % 4x3 matrix of switch times
shaped_slew_delta_mat = slew_base_deltas * amp_shaper;    % 4x3 matrix of step changes

slew_times_flat  = shaped_slew_times_mat(:);
slew_deltas_flat = shaped_slew_delta_mat(:);

% --- PHASE 3: Hoist Down (Positive: +1, starts after Slew finishes + t_wait) ---
t_phase3_start   = max(slew_times_flat) + t_wait; 

t_dn_decel_start = d_hoist_down / v_hoist;             % 18.3333 s
t_dn_total       = t_dn_decel_start + t_hoist_accel;   % 19.3333 s

phase3_rel_times = [0.0; t_hoist_accel; t_dn_decel_start; t_dn_total];
phase3_abs_times = t_phase3_start + phase3_rel_times;
phase3_hoist_deltas = [
     1;  % 0 -> +1 (Start hoist down: positive)
    -1;  % +1 -> 0 (Reach cruise after 1.0 s)
    -1;  % 0 -> -1 (Start decelerating hoist down)
     1   % -1 -> 0 (Stop)
];

% =========================================================================
% --- 3. Combine All Multi-Axis Step Transitions ---
% =========================================================================
all_times = [phase1_times; slew_times_flat; phase3_abs_times];

trolley_deltas = zeros(size(all_times));
slew_deltas    = [zeros(size(phase1_times)); slew_deltas_flat;             zeros(size(phase3_abs_times))];
hoist_deltas   = [phase1_hoist_deltas;       zeros(size(slew_times_flat)); phase3_hoist_deltas];

% =========================================================================
% --- 4. Sort Chronologically, Merge Coincident Timestamps & Integrate ---
% =========================================================================
all_times_rounded = round(all_times, 6);
[time_s, ~, ic]   = unique(all_times_rounded);

trolley_net_deltas = accumarray(ic, trolley_deltas);
slew_net_deltas    = accumarray(ic, slew_deltas);
hoist_net_deltas   = accumarray(ic, hoist_deltas);

% Convert step changes (deltas) into piecewise-constant normalized states (-1, 0, 1)
trolley_accel = sign(round(cumsum(trolley_net_deltas), 6));
slew_accel    = sign(round(cumsum(slew_net_deltas), 6));
hoist_accel   = sign(round(cumsum(hoist_net_deltas), 6));

% =========================================================================
% --- 5. Ensure Final Row Ends Cleanly at Zero Acceleration for All Axes ---
% =========================================================================
if slew_accel(end) ~= 0 || hoist_accel(end) ~= 0 || trolley_accel(end) ~= 0
    time_s(end+1, 1)        = time_s(end) + 0.1;
    trolley_accel(end+1, 1) = 0;
    slew_accel(end+1, 1)    = 0;
    hoist_accel(end+1, 1)   = 0;
end

% =========================================================================
% --- 6. Export to Excel with Exact Lab Headers (Normalized -1, 0, 1) ---
% =========================================================================
T = table(time_s, trolley_accel, slew_accel, hoist_accel);
writetable(T, 'shaped_trajectory.xlsx', 'Sheet', 1);
disp(T);
disp('Successfully generated normalized (-1, 0, 1) shaped_trajectory.xlsx!');