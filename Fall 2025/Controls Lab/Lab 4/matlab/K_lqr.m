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

tau = 1.263e-3;

Lm = R*tau;

g=-9.8;
% ---- common denominator ----
delta = J0*J2 - (m2^2)*(L1^2)*(l2^2);

% ---- unscaled B-tilde (pure mechanics) ----
B31_t =  J2/delta;
B41_t =  m2*L1*l2/delta;

% ---- A entries (use B-tilde inside A for the Km^2/R terms) ----
A31 = 0;
A32 =  g*m2^2*l2^2*L1/delta;
A33 = -b1*J2/delta   - (Km^2/R)*B31_t;
A34 = -b2*m2*l2*L1/delta;

A41 = 0;
A42 =  g*m2*l2*J0/delta;
A43 = -b1*m2*l2*L1/delta - (Km^2/R)*B41_t;
A44 = -b2*J0/delta;

% --- given state-space ---




A = [0 0 1 0 0;
     0 0 0 1 0;
     A31 A32 A33 A34 B31_t*Km;
     A41 A42 A43 A44 B41_t*Km;
     0   0   -Km/Lm 0 -R/Lm]

B = [0; 0; 0; 0; 1/Lm]


qt1 = 100;
qt2 = 400;
qv1 = 300;
qv2 = 200;
qi = 3;


Q = [qt1 0 0 0 0;
    0 qt2 0 0 0;
    0 0 qv1 0 0;
    0 0 0 qv2 0;
    0 0 0 0 qi];

R = [7];


K = lqr(A,B,Q,R)

Acl = A - B*K;
eig(Acl);   % should all have negative real parts

C = [1 0 0 0 0;
     0 1 0 0 0];
dt = 0.001;
% choose poles ~5x faster than controller
poles = [-50 -55 -60 -65 -70];
L = place(A', C', poles)';

disp(L)

Ad_obs = eye(5) + dt*(A - L*C);
eig(Ad_obs)
