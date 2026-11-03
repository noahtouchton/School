clc; clear;

% =========================================================================
% --- 0. TUNABLE LAB VARIABLES (Adjust Here for Physical Machine) ---
% =========================================================================
t_wait         = 1.0;   % Waiting period (s) between each phase (Hoist -> Slew -> Hoist)
d_slew         = 40.0;  % Target slew angle magnitude (deg) - increase (e.g. 42, 45) if undershooting
slew_speed_pct = -80;   % Slew cruise speed percentage (-80% = negative slew direction)
d_hoist_up     = 550.0; % Hoist UP distance (mm)   [1350 mm -> 800 mm = 550 mm]
d_hoist_down   = 550.0; % Hoist DOWN distance (mm) [800 mm -> 1350 mm = 550 mm]

% =========================================================================
% --- 1. Workspace Parameters & Limits (Table IV) ---
% =========================================================================
dt = 0.05;              % Required sample period: 50 ms
L  = 0.800;             % Effective pendulum suspension length (m)
g  = 9.81;              % Gravitational acceleration (m/s^2)
wn = sqrt(g / L);       % Natural frequency (rad/s)

% Maximum velocities from Table IV
v_max_trolley = 72.6;   % mm/s
v_max_slew    = 32.4;   % deg/s
v_max_hoist   = 30.0;   % mm/s

% =========================================================================
% --- 2. Discrete Input Shaper Formulation ---
% =========================================================================
% Unity-Magnitude ZV (UMZV) shaper parameters
A_shaper   = [1, -1, 1];
tau_shaper = [0, acos(0.5)/wn, acos(-0.5)/wn]; % continuous delays (s)

% Discretize onto the 50 ms time grid
idx_delays    = round(tau_shaper / dt); % sample index offsets
shaper_filter = zeros(max(idx_delays) + 1, 1);
for i = 1:length(A_shaper)
    shaper_filter(idx_delays(i) + 1) = shaper_filter(idx_delays(i) + 1) + A_shaper(i);
end

% Wait block of zeros to insert between phases
n_wait     = round(t_wait / dt);
wait_zeros = zeros(n_wait, 1);

% =========================================================================
% --- 3. Unshaped Sequence Construction & Convolution ---
% =========================================================================

% --- PHASE 1: Hoist Up (Negative: -100%, Trapezoidal with t1 = 1.0 s) ---
accel_time_hoist = 1.0;                                           % 1.0 s ramp up/down
hoist_up_pct     = -100;                                          % Negative lifts hook up
t_cruise_up      = (d_hoist_up / v_max_hoist) - accel_time_hoist; % 17.3333 s
t_total_up       = 2 * accel_time_hoist + t_cruise_up;            % 19.3333 s

t_grid_p1 = (0:dt:t_total_up)';
hoist_p1  = zeros(size(t_grid_p1));

up_ramp_up   = t_grid_p1 <= accel_time_hoist;
up_ramp_down = t_grid_p1 > (accel_time_hoist + t_cruise_up);
up_cruise    = ~up_ramp_up & ~up_ramp_down;

hoist_p1(up_ramp_up)   = hoist_up_pct * (t_grid_p1(up_ramp_up) / accel_time_hoist);
hoist_p1(up_cruise)    = hoist_up_pct;
hoist_p1(up_ramp_down) = hoist_up_pct * (1 - (t_grid_p1(up_ramp_down) - (accel_time_hoist + t_cruise_up)) / accel_time_hoist);
slew_p1                = zeros(size(t_grid_p1));

% --- PHASE 2: Slew Traversal (Negative: slew_speed_pct, Tunable d_slew) ---
accel_time_slew = 1.0;                                              % 1.0 s ramp up/down
v_cruise_slew   = abs(slew_speed_pct / 100) * v_max_slew;           % deg/s magnitude
t_cruise_slew   = (abs(d_slew) / v_cruise_slew) - accel_time_slew;  % exact trapezoidal cruise time
t_total_slew    = 2 * accel_time_slew + t_cruise_slew;

t_grid_p2          = (dt:dt:t_total_slew)';
v_unshaped_slew_p2 = zeros(size(t_grid_p2));

slew_ramp_up   = t_grid_p2 <= accel_time_slew;
slew_ramp_down = t_grid_p2 > (accel_time_slew + t_cruise_slew);
slew_cruise    = ~slew_ramp_up & ~slew_ramp_down;

v_unshaped_slew_p2(slew_ramp_up)   = slew_speed_pct * (t_grid_p2(slew_ramp_up) / accel_time_slew);
v_unshaped_slew_p2(slew_cruise)    = slew_speed_pct;
v_unshaped_slew_p2(slew_ramp_down) = slew_speed_pct * (1 - (t_grid_p2(slew_ramp_down) - (accel_time_slew + t_cruise_slew)) / accel_time_slew);

% Convolve Phase 2 slew with discrete UMZV shaper
slew_p2  = conv(v_unshaped_slew_p2(:), shaper_filter(:));
hoist_p2 = zeros(size(slew_p2));

% --- PHASE 3: Hoist Down (Positive: +100%, Trapezoidal with t1 = 1.0 s) ---
hoist_dn_pct = 100;                                             % Positive lowers hook down
t_cruise_dn  = (d_hoist_down / v_max_hoist) - accel_time_hoist; % 17.3333 s
t_total_dn   = 2 * accel_time_hoist + t_cruise_dn;              % 19.3333 s

t_grid_p3 = (dt:dt:t_total_dn)';
hoist_p3  = zeros(size(t_grid_p3));

dn_ramp_up   = t_grid_p3 <= accel_time_hoist;
dn_ramp_down = t_grid_p3 > (accel_time_hoist + t_cruise_dn);
dn_cruise    = ~dn_ramp_up & ~dn_ramp_down;

hoist_p3(dn_ramp_up)   = hoist_dn_pct * (t_grid_p3(dn_ramp_up) / accel_time_hoist);
hoist_p3(dn_cruise)    = hoist_dn_pct;
hoist_p3(dn_ramp_down) = hoist_dn_pct * (1 - (t_grid_p3(dn_ramp_down) - (accel_time_hoist + t_cruise_dn)) / accel_time_hoist);
slew_p3                = zeros(size(t_grid_p3));

% =========================================================================
% --- 4. Combine Sequential Phases with Wait Periods ---
% =========================================================================
hoist_pct   = [hoist_p1; wait_zeros; hoist_p2; wait_zeros; hoist_p3];
slew_pct    = [slew_p1;  wait_zeros; slew_p2;  wait_zeros; slew_p3];
trolley_pct = zeros(size(slew_pct));

% =========================================================================
% --- 5. Bound Constraints and Zero-Termination ---
% =========================================================================
% Clamp to allowed [-100, 100] percentage bounds
slew_pct  = max(-100, min(100, slew_pct));
hoist_pct = max(-100, min(100, hoist_pct));

% Align time vector to exact number of rows
N      = length(slew_pct);
time_s = (0:dt:(N-1)*dt)';

% Ensure trajectory terminates at zero velocity across all axes
if trolley_pct(end) ~= 0 || slew_pct(end) ~= 0 || hoist_pct(end) ~= 0
    time_s(end+1, 1)      = time_s(end) + dt;
    trolley_pct(end+1, 1) = 0;
    slew_pct(end+1, 1)    = 0;
    hoist_pct(end+1, 1)   = 0;
end

% =========================================================================
% --- 6. Export to Excel (Table V Format) ---
% =========================================================================
T = table(time_s, trolley_pct, slew_pct, hoist_pct);
writetable(T, 'continuous_trajectory.xlsx', 'Sheet', 1);
disp('Saved continuous_trajectory.xlsx successfully.');