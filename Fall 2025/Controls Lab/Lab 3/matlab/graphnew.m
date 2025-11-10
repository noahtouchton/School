clear; clc; close all;
R = 16 ;
Km = 0.525 ;
L1 = 0.1231 ;
m2 = 0.00488 ;
J2 = 0.0001374; 
l2 = 0.1472 ;

b1 = 0.0146; 
J0 = 0.00224;

b2 = 4.72*10^-5 ;

g=-9.8;
% ---- common denominator ----
delta = J0*J2 - (m2^2)*(L1^2)*(l2^2);

% ---- unscaled B-tilde (pure mechanics) ----
B31_t =  J2/delta;
B41_t =  m2*L1*l2/delta;

% ---- A entries (use B-tilde inside A for the Km^2/R terms) ----
A32 =  g*m2^2*l2^2*L1/delta;
A33 = -b1*J2/delta   - (Km^2/R)*B31_t;
A34 = -b2*m2*l2*L1/delta;

A42 =  g*m2*l2*J0/delta;
A43 = -b1*m2*l2*L1/delta - (Km^2/R)*B41_t;
A44 = -b2*J0/delta;

% --- given state-space ---
A = [ 0     0      1        0;
      0     0      0        1;
      0   0.459  -0.365  -0.000587;
      0  39.615   0.365   -0.344 ];

A = [ 0     0      1        0;
      0     0      0        1;
      0   A32  A33  A34;
      0  A42   A43   A44 ]


B = [0; 0; 25.0; 12.44];
B = [0; 0; (Km/R)*B31_t; (Km/R)*B41_t]

% Outputs: 1) base angle theta, 2) absolute pendulum angle theta+phi
C = [1 0 0 0;
     1 1 0 0];
D = zeros(2,1);

sys = ss(A,B,C,D);

% --- impulse response ---
t = 0:0.001:10;                 % 5 s window; adjust if you need longer
[y,t,~] = impulse(sys, t);     % unit-area impulse at input

theta   = y(:,1);
phi_abs = y(:,2);

% --- Figure 1: base angle ---
figure('Color','w');
plot(t, theta, 'LineWidth',1.6);
grid on; xlabel('Time (s)'); ylabel('\theta (rad)');
title('Impulse Response — Platen (Base) Angle');

% --- Figure 2: pendulum absolute angle ---
figure('Color','w');
plot(t, phi_abs, 'LineWidth',1.6);
grid on; xlabel('Time (s)'); ylabel('\theta+\phi (rad)');
title('Impulse Response — Pendulum Absolute Angle');
