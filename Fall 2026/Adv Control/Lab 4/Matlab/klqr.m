clc; clear;

%CONSTANTS
L  = 1.0;               %Length
mc = 3;                 %Mass of Cart
mp = 1;                 %Mass of Payload
b  = 0.1;               %Damping factor
g  = 9.81;              %Gravity


t_max = 5 * pi/180;
w_max = 3.14;
x_max = 1;
v_max = 100;

u_max = 1;

q11 = 1/t_max^2;
q22 = 1/w_max^2;
q33 = 1/x_max^2;
q44 = 1/v_max^2;
r11 = 1/u_max^2;

Q = diag([q11, q22, q33, q44]);
R = [r11];

A_acc = [0 1 0 0;
    g/L -b/(mp*L) 0 0;
    0 0 0 1;
    0 0 0 0];
B_acc = [0; 1/L; 0; 1];

K_acc = lqr(A_acc, B_acc, Q, R);

disp(K_acc);

A_for = [0 1 0 0;
         g/(L*mc)*(mc+mp) -b/(mp*L) 0 0;
         0 0 0 1;
         mp*g/mc 0 0 0];
B_for = [0; 1/(L*mc); 0; 1/mc];

K_for = lqr(A_for, B_for, Q, R);


disp(K_for);

