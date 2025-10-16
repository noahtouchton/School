%% === Inputs (fill these from Lab 1 and your spec) ===
K   = 1.163;      % process DC gain from Lab 1
tau = 0.02995;      % process time constant (s) from Lab 1

Ts_target = 0.30;   % desired 2% settling time (s)
eps_ss    = 0.01;   % allowed steady-state error (e.g. 1% = 0.01)

%% === Design targets translated ===
tau_cl = Ts_target/4;                 % first-order: Ts ≈ 4*tau_cl

% Minimum Kp to satisfy steady-state error bound: e_inf = 1/(1+K*Kp) <= eps_ss
Kp_min = (1/eps_ss - 1)/K;

% You can choose Kp at or above this minimum (add margin if you like)
Kp = Kp_min;                           % choose equal to the minimum for now

% Solve Kd from tau_cl = (tau + K*Kd) / (1 + K*Kp)
Kd = (tau_cl*(1 + K*Kp) - tau) / K;

% Map to lab notation
k1 = Kp;   % proportional on position error
k2 = Kd;   % derivative on position error
fprintf('Designed gains:\n   k1 (Kp) = %.4f\n   k2 (Kd) = %.4f\n', k1, k2);

%% === Quick analytic checks ===
e_inf = 1/(1 + K*Kp);                  % predicted steady-state error
tau_cl_check = (tau + K*Kd)/(1 + K*Kp);% predicted closed-loop time constant
Ts_pred = 4*tau_cl_check;              % predicted 2%% settling time

fprintf('Predicted performance:\n   e_inf = %.4f (%.2f%%)\n   Ts(2%%) ≈ %.4f s\n', ...
    e_inf, 100*e_inf, Ts_pred);

%% === (Optional) closed-loop transfer and step check ===
% For PD on first-order plant, closed-loop is first order:
% T(s) = (K*(Kp + Kd*s)) / (tau*s + 1 + K*(Kp + Kd*s))
s = tf('s');
G  = K/(tau*s + 1);
C  = Kp + Kd*s;
Tcl = feedback(C*G, 1);  % unity-feedback

figure; step(Tcl, 1.0); grid on;
title('Closed-loop Step Response (PD on first-order plant)');
info = stepinfo(Tcl);
disp(info);

%% === (Optional) 0.25 Hz square tracking, for curiosity ===
Adeg = 111;                 % reference amplitude in degrees, if relevant
Aref = Adeg*pi/180;         % radians for simulation
f = 0.25; T = 1/f;
t = 0:0.002:3*T;            % simulate ~3 periods
r = Aref * square(2*pi*f*t);

[y, ~] = lsim(Tcl, r, t);
figure; plot(t, r, 'LineWidth', 1.1); hold on;
plot(t, y, 'LineWidth', 1.1); grid on;
xlabel('Time (s)'); ylabel('\theta (rad)');
legend('ref','output'); title('0.25 Hz Square Tracking (PD)');