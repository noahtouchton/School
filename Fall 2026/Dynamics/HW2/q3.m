clc; clear;

psi = 120;
theta = 60;
phi = 45;

L1 = 2;
L2 = 7;
L3 = L2;

r = rotz(psi)*[L1; 0; 0] + rotz(psi)*roty(-theta)*[L2; 0; 0] + rotz(psi)*roty(-theta)*roty(phi)*[L3; 0; 0];

disp(r);
