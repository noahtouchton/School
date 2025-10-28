load question02.mat -ascii
time = question02(:,1);
pos  = question02(:,2);

plot(time,pos)
title('Position vs Time')

% find natural frequency from oscillation period
[pks, locs] = findpeaks(pos, time);
T = mean(diff(locs));         % average period between peaks
omega_d = 2*pi/T;             % damped natural frequency

m = 200;       % kg
c = 1750;      % Ns/m
% critical damping
omega_n = sqrt(omega_d^2 + (c/(2*m))^2);  % natural frequency (undamped)
% two springs (left & right) each with same k
k_single = (m*omega_n^2)/2;               % stiffness per spring
fprintf('k per spring = %.2f N/m\n', k_single);