% Define the grid
[X, Y] = meshgrid(-3:0.2:4, -3:0.2:3);

% Define the functions
Z1 = 0.5 * (X.^2 + Y.^2) - X;
Z2 = 0.5 * Y.^2 - X;
Z3 = 0.5 * X.^2 - X;
Z4 = 0.5 * (X.^2 - Y.^2) - X;

% Create a figure with 4 subplots
figure;

% Plot i
subplot(2,2,1);
contour(X, Y, Z1, 20);
hold on; plot(1, 0, 'r*', 'MarkerSize', 8); % Mark the minimum
title('i) f(x,y) = 0.5(x^2 + y^2) - x');
xlabel('x'); ylabel('y');

% Plot ii
subplot(2,2,2);
contour(X, Y, Z2, 20);
title('ii) f(x,y) = 0.5y^2 - x');
xlabel('x'); ylabel('y');

% Plot iii
subplot(2,2,3);
contour(X, Y, Z3, 20);
hold on; xline(1, 'r--', 'LineWidth', 1.5); % Mark the line of minima
title('iii) f(x,y) = 0.5x^2 - x');
xlabel('x'); ylabel('y');

% Plot iv
subplot(2,2,4);
contour(X, Y, Z4, 20);
hold on; plot(1, 0, 'ro', 'MarkerSize', 8); % Mark the saddle point
title('iv) f(x,y) = 0.5(x^2 - y^2) - x');
xlabel('x'); ylabel('y');