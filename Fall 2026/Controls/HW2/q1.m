clc; clear;
A = [-3 2 1; 1 -4 -1; -1 -2 -5];
B = [1 0; 0 2; 1 1];
T = [1 1 0; 1 0 1; 0 1 1];

A_hat = T*A*T^-1;
B_hat = T*B;

disp(A_hat);
disp(B_hat);

x0 = [1;0;1];
x0_hat = T*x0;

disp(x0_hat);

syms t

x_hat_t = expm(A_hat*t) * x0_hat; 

x_t = T^-1 * x_hat_t;

disp(x_t);