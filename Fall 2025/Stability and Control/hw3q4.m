clc; clear; close all;

%% Parameters
m1 = 2;  m2 = 2;           % kg
c1 = 1;  c2 = 1;           % N*s/m
k0 = 20;                   % linear part (all three)
kl = 100; kr = 200;        % cubic hardening (left/right) -- only for nonlinear

%% Linearized state-space (x = [y1; y2; y1dot; y2dot])
A = [ 0  0  1   0;
      0  0  0   1;
     -2*k0/m1  k0/m1  -c1/m1  0;
      k0/m2  -2*k0/m2  0     -c2/m2 ];
B = [0;0;1/m1;0];
C = [1 0 0 0;   % output y1 (left cart)
     0 1 0 0]; % (and y2 if you want)
D = zeros(2,1);
sys_lin = ss(A,B,C,D);

%% (c) Eigenvalues / eigenvectors
[V,Dlam] = eig(A);
lambda = diag(Dlam);
disp('Eigenvalues (linearized A):'); disp(lambda);
disp('Eigenvectors (columns of V):'); disp(V);

% Modal parameters per pole (continuous-time)
sig     = real(lambda);
omega_d = abs(imag(lambda));
omega_n = sqrt(sig.^2 + omega_d.^2);
zeta    = -sig ./ max(omega_n,eps);
t_half  = log(2)./max(abs(sig),eps);
T = table(lambda,omega_n,omega_d,zeta,t_half);
disp('Modal parameters from poles:'); disp(T);

%% Initial condition & time
x0 = [1; 0; 0; 0];     % y1=1, y2=0, y1dot=0, y2dot=0
tspan = [0 10];

%% (d) initial() response of the linear SS (no input)
[y_init, t_init, x_init] = initial(sys_lin, x0, tspan);

%% (e) ode45 on the linear ODE (should match (d))
f_lin = @(t,x) [ x(3);
                 x(4);
                (-2*k0*x(1)+k0*x(2) - c1*x(3))/m1;
                ( k0*x(1)-2*k0*x(2) - c2*x(4))/m2 ];
[ t_ol, x_ol ] = ode45(f_lin, tspan, x0);
y1_ol = x_ol(:,1);

%% (f) ode45 on the nonlinear ODE (hardening left/right; input u=0)
f_nl = @(t,x) [ x(3);
                x(4);
               ( -c1*x(3) ...
                 - k0*x(1) - kl*x(1)^3 ...
                 - k0*(x(1)-x(2)) )/m1;                           % m1
               ( -c2*x(4) ...
                 - k0*x(2) - kr*x(2)^3 ...
                 - k0*(x(2)-x(1)) )/m2 ];                         % m2
[ t_nl, x_nl ] = ode45(f_nl, tspan, x0);
y1_nl = x_nl(:,1);

%% (g) Overlay y1(t) from all three
figure; hold on; grid on;
plot(t_init, y_init(:,1), 'k', 'LineWidth', 1.4);      % initial() linear
plot(t_ol,   y1_ol,      'b--', 'LineWidth', 1.4);     % ode45 linear
plot(t_nl,   y1_nl,      'r', 'LineWidth', 1.4);       % ode45 nonlinear
xlabel('Time (s)'); ylabel('Position y_1 (m)');
legend('linear: initial()', 'linear: ode45', 'nonlinear: ode45', 'Location','best');
title('Left cart response y_1(t): linear vs nonlinear');

%% (Optional) If you want the right-cart traces too
% figure; hold on; grid on;
% plot(t_init, y_init(:,2), 'k', 'LineWidth', 1.4);
% plot(t_ol,   x_ol(:,2),  'b--', 'LineWidth', 1.4);
% plot(t_nl,   x_nl(:,2),  'r', 'LineWidth', 1.4);
% xlabel('Time (s)'); ylabel('Position y_2 (m)');
% legend('linear: initial()', 'linear: ode45', 'nonlinear: ode45');
% title('Right cart response y_2(t)');
