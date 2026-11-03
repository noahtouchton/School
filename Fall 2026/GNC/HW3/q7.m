clc; clear; close('all');

c = [-60 -20];

A = [-1 0;
    0 -1;
    1 1;
    1 0;
    0 1;
    -1 -1];

b = [0;0;500;200;360;-160];

[x_star, fval, exitflag] = linprog(c, A, b);

total_cost = fval + 461200;

fprintf('--- Optimal Solution ---\n');
fprintf('x1 (Seattle -> Denver) : %.0f cars\n', x_star(1));
fprintf('x2 (Seattle -> Miami)  : %.0f cars\n', x_star(2));
fprintf('Seattle -> New York    : %.0f cars\n', 500 - sum(x_star));
fprintf('Chicago -> Denver      : %.0f cars\n', 200 - x_star(1));
fprintf('Chicago -> Miami       : %.0f cars\n', 360 - x_star(2));
fprintf('Chicago -> New York    : %.0f cars\n', sum(x_star) - 160);
fprintf('Minimum Total Cost     : $%.2f\n', total_cost);