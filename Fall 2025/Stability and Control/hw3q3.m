clc; clear; close all;

%% --- parameters ---
k1 = 20; k2 = 20; k3 = 20;     % N/m
m1 = 2;  m2 = 2;               % kg
c1 = 1;  c2 = 1;               % N·s/m

%% --- state-space (x = [y1+y2; y1−y2; y1dot; y2dot]) ---
A = [  0     0      1      1;
       0     0      1     -1;
      -k1/(2*m1) -(k1/2+k2)/m1   -c1/m1      0;
      -k3/(2*m2)  (k3/2+k2)/m2     0      -c2/m2 ];
B = [0; 0; 1/m1; 0];
C = [0.5  0.5  0  0;     % y1
     0.5 -0.5  0  0];    % y2
D = zeros(2,1);
sys = ss(A,B,C,D);

%% --- (b) eigenvalues/eigenvectors ---
[V,Dlam] = eig(A);
lambda = diag(Dlam);
disp('Eigenvalues (λ):'); disp(lambda);
disp('Eigenvectors (columns of V):'); disp(V);

%% --- (c) modal quantities ---
sig     = real(lambda);               % σ
omega_d = abs(imag(lambda));          % damped freq
omega_n = sqrt(sig.^2 + omega_d.^2);  % natural freq
zeta    = -sig ./ omega_n;            % damping ratio
t_half  = log(2) ./ max(abs(sig),eps);% half-amplitude time
T = table(lambda,omega_n,omega_d,zeta,t_half);
disp('Modal parameters:'); disp(T);

%% --- time grid ---
t_end = 25; N = 5001;
t = linspace(0,t_end,N).';
dt = t(2)-t(1);
Fs = 1/dt;                            % Hz

%% --- IMPULSE response + FFT ---
[y_imp, t_imp] = impulse(sys,t);

figure; plot(t_imp,y_imp(:,1),'b',t_imp,y_imp(:,2),'r','LineWidth',1.4);
grid on; xlabel('Time (s)'); ylabel('Position (m)');
legend('y_1 (left)','y_2 (right)','Location','best');
title('Impulse Response');

% FFT (impulse)
Y1 = fft(y_imp(:,1));
Y2 = fft(y_imp(:,2));
f  = (0:N-1)' * Fs/N;
figure;
subplot(2,1,1); plot(f,abs(Y1),'LineWidth',1.1); xlim([0 5]); grid on;
xlabel('Frequency (Hz)'); ylabel('|FFT(y_1)|'); title('FFT of Impulse Response (y_1)');
subplot(2,1,2); plot(f,abs(Y2),'LineWidth',1.1); xlim([0 5]); grid on;
xlabel('Frequency (Hz)'); ylabel('|FFT(y_2)|'); title('FFT of Impulse Response (y_2)');

%% --- STEP response + FFT ---
[y_step, t_step] = step(sys,t);

figure; plot(t_step,y_step(:,1),'b',t_step,y_step(:,2),'r','LineWidth',1.4);
grid on; xlabel('Time (s)'); ylabel('Position (m)');
legend('y_1 (left)','y_2 (right)','Location','best');
title('Step Response');

% FFT (step) – detrend to suppress the large DC component
y1s = detrend(y_step(:,1),'constant');   % remove mean
y2s = detrend(y_step(:,2),'constant');
Y1s = fft(y1s);
Y2s = fft(y2s);

figure;
subplot(2,1,1); plot(f,abs(Y1s),'LineWidth',1.1); xlim([0 5]); grid on;
xlabel('Frequency (Hz)'); ylabel('|FFT(y_1 step, detrend)|');
title('FFT of Step Response (y_1)');
subplot(2,1,2); plot(f,abs(Y2s),'LineWidth',1.1); xlim([0 5]); grid on;
xlabel('Frequency (Hz)'); ylabel('|FFT(y_2 step, detrend)|');
title('FFT of Step Response (y_2)');