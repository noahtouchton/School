clear; clc; close all;

% 1. System Matrices
A = [-0.3 -5 10; 
    -0.2 -2  0; 
    0    1  0];
B = [10; 10; 0];
Bw = [-0.4; -0.2; 0];
C = [1 0 0];

% 2. Augmented System for Integral Control (z = [x; x_I])
A_aug = [A, zeros(3,1); 
    C, 0];
B_aug = [B; 0];

% 3. Pole Placement (from Part b)
% Note: 'place' requires distinct poles, so we slightly perturb the repeated pole at -2
poles = [-2, -2.0001, sqrt(2)*(-1 + 1j), sqrt(2)*(-1 - 1j)];
K_aug = place(A_aug, B_aug, poles);

k = K_aug(1:3)';
k_I = K_aug(4);

fprintf('Controller Gains:\n');
fprintf('k1 = %.4f, k2 = %.4f, k3 = %.4f, kI = %.4f\n', k(1), k(2), k(3), k_I);

% 4. Closed-Loop System Matrices
A_cl = A_aug - B_aug * K_aug;
B_vd = [0; 0; 0; -1];
B_w_aug = [Bw; 0];
C_cl = [1 0 0 0]; % Output is horizontal velocity v = x1

% Create state-space objects
sys_vd = ss(A_cl, B_vd, C_cl, 0);
sys_w  = ss(A_cl, B_w_aug, C_cl, 0);

% 5. Simulation Setup (vd = 1, zero initial conditions)
t = 0:0.01:10;
vd = ones(size(t));

% Case 1: Zero disturbance (w = 0)
v_w0 = lsim(sys_vd, vd, t);

% Case 2: Nonzero step disturbance (w = 10)
w = 10 * ones(size(t));
v_w10 = lsim(sys_vd, vd, t) + lsim(sys_w, w, t);

% 6. Plotting v(t) vs. t
figure('Color', 'w');
plot(t, v_w0, 'b-', 'LineWidth', 2); hold on;
plot(t, v_w10, 'r--', 'LineWidth', 2);
yline(1, 'k:', 'LineWidth', 1.2);
grid on;

title('Closed-Loop Horizontal Velocity Response v(t) for v_d = 1', 'FontSize', 12);
xlabel('Time t (s)', 'FontSize', 11);
ylabel('Horizontal Velocity v(t)', 'FontSize', 11);
legend('Zero Disturbance (w = 0)', 'Headwind Disturbance (w = 10)', 'Target Velocity (v_d = 1)', 'Location', 'best');