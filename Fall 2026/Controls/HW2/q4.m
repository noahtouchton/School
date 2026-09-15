clc; clear;


syms theta1 theta2 theta3 theta4 L1 L2 L3 v1 v2 real


x1 = cos(theta1) * (L1*cos(theta2) + L2*cos(theta2+theta3) + L3*cos(theta2+theta3+theta4));
x2 = sin(theta1) * (L1*cos(theta2) + L2*cos(theta2+theta3) + L3*cos(theta2+theta3+theta4));
x3 = L1*sin(theta2) + L2*sin(theta2+theta3) + L3*sin(theta2+theta3+theta4);


X = [x1; x2; x3];

% Substitute L1 = L2 = L3 = 1 for the rest of the problem
X_sub = subs(X, [L1, L2, L3], [1, 1, 1]);


J_analytical = jacobian(X_sub, [theta1, theta2, theta3, theta4]);


theta0 = [0, 0, pi/2, 0];
J = subs(J_analytical, [theta1, theta2, theta3, theta4], theta0);
%J = double(J)




J_plus = J.' / (J*J.');
disp(J_plus);

omega = J_plus * [v1; v2; 0];

disp(omega);

Tau = J.' * [0;0;50];

disp(Tau);

theta_star = [0, pi/4, pi/4, pi/4];

J_star = subs(J_analytical, [theta1, theta2, theta3, theta4], theta_star);

disp(J_star);
