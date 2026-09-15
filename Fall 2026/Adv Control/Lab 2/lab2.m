clc; clear

% --- 1. Define Parameters (from Table 1) ---
m_h = 0.407; % Hook mass (kg)
m_p = 0.569; % Payload mass (kg)
L1  = 0.600; % Length 1 (m)
L2  = 0.200; % Length 2 (m)
g   = 9.81;  % Acceleration due to gravity (m/s^2)

% --- 2. Calculate Intermediate Variables ---
% Calculate the mass ratio, R
R = m_p / m_h;

% Calculate beta
% Breaking it down into two terms for cleaner code
beta_term1 = (1 + R)^2 * (1/L1 + 1/L2)^2;
beta_term2 = 4 * ((1 + R) / (L1 * L2));
beta = sqrt(beta_term1 - beta_term2);

% --- 3. Calculate Natural Frequencies ---
% The formula uses a minus-plus sign. We will calculate both w1 (-) and w2 (+).
freq_term = (1 + R) * (1/L1 + 1/L2);

w1 = sqrt((g / 2) * (freq_term - beta));
w2 = sqrt((g / 2) * (freq_term + beta));

% --- 4. Display the Results ---
fprintf('--- Results ---\n');
fprintf('Mass Ratio (R): %.4f\n', R);
fprintf('Beta: %.4f\n', beta);
fprintf('Natural Frequency w1: %.4f rad/s\n', w1);
fprintf('Natural Frequency w2: %.4f rad/s\n', w2);


T1 = 2*pi/w1;
T2 = 2*pi/w2;

% calcualte one mode zv shaper %

td1 = T1/2;
td2 = T2/2;

fprintf("td1: %.4f s\n", td1);
fprintf("td2: %.4f s\n", td2);

% ZVD %

tzvd1 = pi/w1;
tzvd2 = 2*pi/w1;

fprintf("tzvd1: %.4f s\n", tzvd1);
fprintf("tzvd2: %.4f s\n", tzvd2);


% 2 mode ZV %

% --- Two-Mode ZV Shaper (Convolution of w1 and w2 ZV shapers) ---
% Define the individual ZV shapers
A_zv1 = [0.5, 0.5];
t_zv1 = [0, td1];

A_zv2 = [0.5, 0.5];
t_zv2 = [0, td2];

% Initialize arrays for the 4 convolved impulses
t_conv_unsorted = zeros(1, 4);
A_conv_unsorted = zeros(1, 4);

% Convolve by multiplying amplitudes and adding times
idx = 1;
for i = 1:length(A_zv1)
    for j = 1:length(A_zv2)
        A_conv_unsorted(idx) = A_zv1(i) * A_zv2(j);
        t_conv_unsorted(idx) = t_zv1(i) + t_zv2(j);
        idx = idx + 1;
    end
end

% Sort the convolved impulses chronologically
[t_two_mode, sort_idx] = sort(t_conv_unsorted);
A_two_mode = A_conv_unsorted(sort_idx);

fprintf('\n--- Two-Mode ZV Shaper (Convolved) ---\n');
for i = 1:length(t_two_mode)
    % Multiplied by 100 to format the amplitude as a percentage for the GUI
    fprintf('Impulse %d: Amp = %.2f, Time = %.4f s\n', i, A_two_mode(i)*100, t_two_mode(i));
end




