clc; clear;

global Ap A P;

v_step = 0.2;      
step1 = v_step;
step2 = -v_step;
t_on = 1;
t_off = 5;
t_sim = 10;

g=9.81;
L=1.0;
wn = sqrt(g/L);
wm=wn;
zn = 0.001;

zm = 0.8; % desired value of zeta

Ap = [0 1; -wn^2 -2*zn*wn];
Bp = [0;1];
Cp = eye(2);
Dp = [0;0];

A = [0 1; -wm^2 -2*zm*wm];
B = [0;1];
C = eye(2);
D = [0;0];

Q = eye(2);

syms p11 p12 p22;

P_sym = [p11 p12; p12 p22];

eqn = (A' * P_sym + P_sym * A == -Q);

solution = solve(eqn);

% Convert the symbolic fractions to double-precision numbers
p11_num = double(solution.p11);
p12_num = double(solution.p12);
p22_num = double(solution.p22);

% Assemble the numeric P matrix
P = [p11_num, p12_num; 
    p12_num, p22_num];

disp(P);

sim('student_model_ref');
% --- Format Data for Animation ---
% 1. Extract raw numeric data regardless of Simulink's save format
if isa(times, 'timeseries')
    t_val = times.Time;
else
    t_val = double(times(:));
end

if isa(x, 'timeseries')
    x_val = squeeze(double(x.Data));
    % If squeeze oriented the matrix sideways (2xN), transpose to Nx2
    if size(x_val, 2) ~= 2 && size(x_val, 1) == 2
        x_val = x_val'; 
    end
else
    x_val = double(x);
end

if isa(u, 'timeseries')
    u_val = double(squeeze(u.Data));
else
    u_val = double(u(:));
end

t_val = t_val(:);
u_val = u_val(:);

% 2. Extract payload angle state (column 2)
x2 = x_val(:, 2);

% 3. Calculate Trolley Position via integration
trolleyPos = zeros(length(t_val), 1);
for aa = 2:length(t_val)
    trolleyPos(aa) = trapz(t_val(aa-1:aa), u_val(aa-1:aa)) + trolleyPos(aa-1);
end

% 4. Create standard 2D arrays for the animation blocks
deflAngle = timeseries(x2/L, t_val);
xt = timeseries(trolleyPos, t_val);

% --- Run the Animation ---
sim('student_model_animation');


% --- Pretty Print GUI Variables ---
fprintf('\n=======================================\n');
fprintf('  CRANE GUI INPUTS FOR ZETA_M = %.2f\n', zm);
fprintf('=======================================\n');
fprintf('Axis:      0       0:trolley or Y else:bridge or X\n');
fprintf('Feedback:  2       1:trolley velocity 2:payload position\n\n');

fprintf('P11: %9.3f       Ap11: %9.3f\n', P(1,1), Ap(1,1));
fprintf('P12: %9.3f       Ap12: %9.3f\n', P(1,2), Ap(1,2));
fprintf('P22: %9.3f       Ap21: %9.3f\n', P(2,2), Ap(2,1));
fprintf('A11: %9.3f       Ap22: %9.3f\n', A(1,1), Ap(2,2));
fprintf('A12: %9.3f       Bp1:  %9.3f\n', A(1,2), Bp(1));
fprintf('A21: %9.3f       Bp2:  %9.3f\n', A(2,1), Bp(2));
fprintf('A22: %9.3f\n', A(2,2));
fprintf('B1:  %9.3f\n', B(1));
fprintf('B2:  %9.3f\n', B(2));
fprintf('=======================================\n\n');