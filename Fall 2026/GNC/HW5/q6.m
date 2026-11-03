clear; clc; close all;

% Coordinates (in degrees and radians)
theta_0 = deg2rad(35.7);          % Tokyo latitude (35.7 N)
theta_f = deg2rad(40.7);          % New York latitude (40.7 N)
lon_Tokyo = 139.7;                % Tokyo longitude (139.7 E)
phi_f_deg = (180 - 139.7) + (180 - 73.8); % Total eastward travel = 146.5 deg
phi_f = deg2rad(phi_f_deg);

% Solve for alpha and theta_m as s_f -> infinity (theta(phi_f) = theta_f)
tan_alpha = (-tan(theta_0) * sin(phi_f)) / (tan(theta_f) - tan(theta_0) * cos(phi_f));
alpha = atan(tan_alpha);          % alpha = -0.2566 rad (-14.70 deg)
theta_m = atan(-tan(theta_0) / sin(alpha)); % theta_m = 1.2312 rad (70.54 deg)

% Longitude array from phi = 0 to phi_f
phi = linspace(0, phi_f, 500);
phi_deg = rad2deg(phi);

% (b) Minimum distance path theta(phi)
theta = atan(tan(theta_m) * sin(phi - alpha));
theta_deg = rad2deg(theta);

% (c) Optimal heading angle beta(phi)
% Using d(theta)/d(phi) = tan(beta)*cos(theta) to preserve the sign of beta
% after passing the northernmost apex at theta_m:
cos_beta = cos(theta_m) ./ cos(theta);
sin_beta = sin(theta_m) .* cos(phi - alpha);
beta = atan2(sin_beta, cos_beta);
beta_deg = rad2deg(beta);

% Plotting
figure('Position', [100, 100, 800, 600]);

% Subplot (b): Latitude vs. Longitude
subplot(2,1,1);
plot(phi_deg, theta_deg, 'b-', 'LineWidth', 2); hold on;
plot(0, rad2deg(theta_0), 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 7);
plot(phi_f_deg, rad2deg(theta_f), 'go', 'MarkerFaceColor', 'g', 'MarkerSize', 7);
plot(90 + rad2deg(alpha), rad2deg(theta_m), 'ks', 'MarkerFaceColor', 'k', 'MarkerSize', 7);
grid on;
xlabel('Relative Longitude \phi (deg East of Tokyo)');
ylabel('Latitude \theta (deg N)');
title('(b) Minimum Distance Path \theta vs. \phi (Tokyo to New York)');
legend('Great Circle Path', 'Tokyo (\phi = 0^\circ)', 'New York (\phi_f = 146.5^\circ)', ...
    sprintf('Max Lat \\theta_m = %.1f^\\circ', rad2deg(theta_m)), 'Location', 'South');

% Subplot (c): Heading Angle vs. Longitude
subplot(2,1,2);
plot(phi_deg, beta_deg, 'r-', 'LineWidth', 2); hold on;
yline(0, 'k--', 'Due East (\beta = 0^\circ)');
grid on;
xlabel('Relative Longitude \phi (deg East of Tokyo)');
ylabel('Heading Angle \beta (deg from East)');
title('(c) Optimal Heading Angle \beta vs. Longitude \phi');