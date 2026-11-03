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
wm=wn*1.35;
zn = 0.001;

zm = 0.5; % desired value of zeta

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