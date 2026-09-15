clc; clear;

theta1 = 120;
theta2 = 60;

r0 = rotz(0)*[0; 10; 0] + rotz(0)*roty(0)*[0; 0; -10];
r = rotz(theta1)*[0; 10; 0] + rotz(theta1)*roty(theta2)*[0; 0; -10];

dr = r-r0;

disp(dr);