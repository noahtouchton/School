% Define parameters
p = 1;
q = 2;

% Create grid
[Y, Z] = meshgrid(linspace(-1, 1, 400), linspace(-0.5, 2.5, 400));

% Calculate function values
F = (Z - p * Y.^2) .* (Z - q * Y.^2);

% Create figure
figure('Position', [100, 100, 800, 600]);
hold on;
grid on;

% Plot surface
surf(Y, Z, F, 'EdgeColor', 'none', 'FaceAlpha', 0.8);
colormap('parula'); 
clim([-0.2, 1.0]); % Use caxis([-0.2, 1.0]) if on an older MATLAB version

% Add a contour plot on the bottom floor to show the "valley"
[~, hContour] = contourf(Y, Z, F, linspace(-0.25, 0, 10));
hContour.ContourZLevel = -0.5;

% Plot the parabolic path z = 1.5 * y^2 (m = 1.5, between p=1 and q=2)
m = 1.5;
y_path = linspace(-0.8, 0.8, 100);
z_path = m * y_path.^2;
f_path = (z_path - p * y_path.^2) .* (z_path - q * y_path.^2);
plot3(y_path, z_path, f_path, 'r', 'LineWidth', 3, 'DisplayName', 'Path z = 1.5y^2');

% Plot a straight line path z = y to show it goes positive
y_line = linspace(-0.8, 0.8, 100);
z_line = y_line;
f_line = (z_line - p * y_line.^2) .* (z_line - q * y_line.^2);
plot3(y_line, z_line, f_line, 'w--', 'LineWidth', 2, 'DisplayName', 'Path z = y');

% Mark the origin
plot3(0, 0, 0, 'ko', 'MarkerSize', 10, 'MarkerFaceColor', 'k', 'DisplayName', 'Origin (0,0,0)');

% Formatting, Labels, and View
xlabel('Y', 'FontWeight', 'bold');
ylabel('Z', 'FontWeight', 'bold');
zlabel('f(Y, Z)', 'FontWeight', 'bold');
zlim([-0.5, 2]);
title('Plot of f(y,z) = (z - y^2)(z - 2y^2)', 'FontSize', 14);
view(-45, 20);
legend('Location', 'best');

hold off;