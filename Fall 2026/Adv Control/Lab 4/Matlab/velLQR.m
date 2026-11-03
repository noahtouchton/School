clc; clear;

%CONSTANTS
L  = 2.009;               %Length
mc = 3;                 %Mass of Cart
mp = 0.4781;                 %Mass of Payload
b  = 0.0513;               %Damping factor
g  = 9.81;              %Gravity

int_max = 5;
t_max = 5 * pi/180;
w_max = 2;
x_max = 1/4;
v_max = 100;

u_max = 0.1;

q11 = 1/int_max^2;
q22 = 1/t_max^2;
q33 = 1/x_max^2;
r11 = 1/u_max^2;

Q = diag([q11, q22, q33]);
R = [r11];

A = [0 1 0;
     g/L -b/(mp*L^2) 0;
     0 0 0];

B = [0; 1/L; 1];

K = lqr(A,B,Q,R);

disp(K);