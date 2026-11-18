clear; clc;

%% 1. Define System Matrices & Desired Specifications
A = [-10  0 -10  0;
    0 -1  10  0;
    0 -1  -1  0;
    1  0   0  0];

B = [20  3;
    0 -3;
    0  0;
    0  0];

beta_d = 0.1;  % Desired sideslip angle (rad)
phi_d  = 0.2;  % Desired bank/roll angle (rad)
poles  = [-10, -15, -10 + 10j, -10 - 10j];

fprintf('=== STEP 1: System Controllability Check ===\n');
Co = ctrb(A, B);
fprintf('Rank of Controllability Matrix: %d (System order n = %d)\n\n', rank(Co), size(A,1));

%% 2. Solve for Steady-State Equilibrium (x_ss and u_ss)
% At steady state (x_dot = 0): A*x_ss + B*u_ss = 0
% From row 4: p_ss = 0
% From row 3: -r_ss - beta_ss = 0  =>  r_ss = -beta_d
p_ss = 0;
r_ss = -beta_d;
x_ss = [p_ss; r_ss; beta_d; phi_d];

% Solve B_top * u_ss = -A_top * x_ss using the first two rows
A_top = A(1:2, :);
B_top = B(1:2, :);
u_ss = -B_top \ (A_top * x_ss);

fprintf('=== STEP 2: Steady-State Equilibrium Values ===\n');
fprintf('Steady-state vector x_ss = [p; r; beta; phi]:\n');
disp(x_ss');
fprintf('Required trim control input u_ss = [delta_a; delta_r]:\n');
disp(u_ss');

%% 3. Pole Placement for Gain Matrix K
% Compute K via MATLAB place()
K = place(A, B, poles);

% Also compute the exact decoupled analytical K for comparison
K_analytical = [0.75, -0.90,  8.05, 7.50;
    0.00, -6.00, 57.00, 0.00];

fprintf('=== STEP 3: Feedback Gain Matrix K (via place) ===\n');
disp(K);

fprintf('Closed-loop matrix A_cl = A - B*K:\n');
A_cl = A - B*K;
disp(A_cl);

%% 4. Compute Reference Feedforward Vector r
% At steady state: u_ss = -K*x_ss + r  =>  r = u_ss + K*x_ss
r_vec = u_ss + K * x_ss;
r_analytical = u_ss + K_analytical * x_ss;

fprintf('=== STEP 4: Reference Feedforward Vector r ===\n');
fprintf('r = u_ss + K*x_ss:\n');
disp(r_vec);

%% 5. Final Verification
fprintf('=== STEP 5: Verification ===\n');
fprintf('Closed-loop eigenvalues (eig(A - B*K)):\n');
disp(eig(A_cl));

x_dot_ss = A_cl * x_ss + B * r_vec;
fprintf('Steady-state derivative check (should be ~0):\n');
disp(x_dot_ss');

% Optional: Display analytical decoupled comparison
fprintf('--- Analytical Decoupled Solution Comparison ---\n');
fprintf('K_analytical:\n'); disp(K_analytical);
fprintf('r_analytical:\n'); disp(r_analytical');