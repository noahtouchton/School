clear; clc; close all;

% 1. Plant Matrices
A = [ 0  1  0   0;
    30  0  0  20;
    0  0  0   1;
    -2  0  0 -10];
B = [0; -25; 0; 10];
C = [40  0  0  0;
    0  0 10  0;
    0  0  0 -5];

% 2. State-Feedback Controller Gain K
p_c = [-4, -5, -6+4j, -6-4j];
K = place(A, B, p_c);

% 3. Full-Order Observer Gain L_full
p_o = [-15, -16, -18, -20];
L_full = place(A', C', p_o)';

% 4. Reduced-Order (1st-Order) Observer Parameters
% Pole at s = -18 => l_r1 = 18
Ca_inv = diag([1/40, 1/10, -1/5]);
Lr = [18, 0, 0];

% 5. Closed-Loop Simulation (Initial Condition Response)
x0 = [0.2; 0; -0.5; 0];     % Initial angle 0.2 rad, cart at -0.5 m
xhat0_full = [0; 0; 0; 0];  % Zero initial observer state
z0_red = 0;                 % Zero initial reduced observer state

t = 0:0.01:4;

% Full-order closed-loop system: [x; x_hat]
A_cl_full = [A, -B*K;
    L_full*C, A - B*K - L_full*C];
sys_full = ss(A_cl_full, zeros(8,1), eye(8), 0);
[~, ~, X_full] = initial(sys_full, [x0; xhat0_full], t);

% Reduced-order closed-loop simulation via ode45
ode_red = @(t, state) reduced_cl_dynamics(state, A, B, C, Ca_inv, K, Lr);
[~, X_red_aug] = ode45(ode_red, t, [x0; z0_red]);
X_red = X_red_aug(:, 1:4);

% 6. Plot Comparison
figure('Color', 'w');
subplot(2,1,1);
plot(t, X_full(:,1), 'b-', 'LineWidth', 1.8); hold on;
plot(t, X_red(:,1), 'r--', 'LineWidth', 1.8);
grid on; ylabel('Pendulum Angle \theta (rad)');
title('Full-Order vs. Reduced-Order Observer Closed-Loop Response');
legend('Full-Order Observer', 'Reduced-Order (1st-Order) Observer');

subplot(2,1,2);
plot(t, X_full(:,3), 'b-', 'LineWidth', 1.8); hold on;
plot(t, X_red(:,3), 'r--', 'LineWidth', 1.8);
grid on; ylabel('Cart Position x_3 (m)'); xlabel('Time (s)');

function dstate = reduced_cl_dynamics(state, A, B, C, Ca_inv, K, Lr)
x = state(1:4);
z = state(5);
y = C * x;
xa = Ca_inv * y;              % [x1; x3; x4]
x2_hat = z + Lr * xa;         % Estimated angular velocity
x_hat = [xa(1); x2_hat; xa(2); xa(3)];
u = -K * x_hat;

dx = A*x + B*u;
dz = -18*z - 7.35*y(1) - 4*y(3) - 25*u;
dstate = [dx; dz];
end