clc; clear;

function parametric_pendulum_analysis()
    wn = 1.0;
    zeta = 0.1;
    tspan = [0 150];

    omega_values = [1.5, 1.7, 1.8, 1.9, 2.2, 2.3, 2.4];

    figure('Color', 'w', 'Position', [100, 100, 1200, 700]);

    for k=1:length(omega_values)
        w = omega_values(k);

        Y0 = [1; 0; 0; 1];

        [t, Y] = ode45(@(t,y) dPhi_dt(t, y, wn, zeta, w), tspan, Y0);

        subplot(2,4,k);
        plot(t, Y(:,1), 'b-', 'LineWidth', 1.1); hold on;
        plot(t, Y(:,2), 'r--', 'LineWidth', 1.0);
        plot(t, Y(:,3), 'g-.', 'LineWidth', 1.0);
        plot(t, Y(:,4), 'k:', 'LineWidth', 1.0);
        grid on;
        title(['\omega = ', num2str(w), ' rad/s']);
        xlabel('Time (s)');
        ylabel('\Phi_{ij}(t)');
        if k==1
            legend('\Phi_{11}', '\Phi_{21}', '\Phi_{12}', '\Phi_{22}', 'Location', 'best');
        end
    end
end

function dy = dPhi_dt(t, y, wn, zeta, w)
    a_t = 0.5*sin(w*t);
    A = [0, 1; -wn^2 * (1+a_t), -2*zeta*wn];
    
    Phi = reshape(y, [2,2]);
    dPhi = A * Phi;
    dy = dPhi(:);
end

parametric_pendulum_analysis();