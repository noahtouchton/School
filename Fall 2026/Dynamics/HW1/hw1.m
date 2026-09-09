clc; clear;

L = 1;
g = 9.81;

fnc = @(t,x) [x(2); (g/L)*cos(x(1))];

x0 = [0.0, 0.0];

t0 = 0; tf = 20;

options = odeset('RelTol',1e-7,'AbsTol',1e-7,'MaxStep',(tf-t0)/10000);

[T,X] = ode45(fnc, [t0 tf], x0, options);

theta = X(:,1);
thetadot = X(:,2);

bottom_idx = find(theta >= pi/2, 1);
timeToBottom = T(bottom_idx);

fprintf('Time to reach the bottom: %.4f seconds\n', timeToBottom);

figure(1)
plot(T,theta)
xlabel('time (s)')
ylabel('theta (rad)')

figure(2)
plot(T,thetadot)
xlabel('time (s)')
ylabel('theta (rad/s)')

