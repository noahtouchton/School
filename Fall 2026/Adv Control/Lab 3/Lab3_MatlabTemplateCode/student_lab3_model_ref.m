%=========================================================
%      ME6404: ADVANCED CONTROLS AND IMPLEMENTATION
%                       
%             LAB 3:  MODEL REFERENCE CONTROL
%==========================================================
clear;clc;

%% Model Reference Control on Payload Oscillation

% **** make a pulse velocity command signal, v
% velocity step size of the portable bridge crane
v_step = 0.2;      % 0.2m/s is 100% speed on the bridge crane
step1 = v_step;
step2 = -v_step;

% times of the velocity input pulse signal, v
t_on = 1;
t_off = 5;
t_sim = 10;         % simulation time

% declaring these variables global
global Ap Bp Cp Dp A B C D P;

% *** Plant (Payload)
g=9.81;
L=1;                      % suspension length
zeta = 0.001;
wn = sqrt(g/L);
Ap = [0 1; -wn^2 -2*zeta*wn];
Bp = [0; 1];
Cp = eye(2);
Dp = [0;0];

% *** Model 
wn_model = wn;
zeta_model = 0.5;
A = [0 1; -wn_model^2 -2*zeta_model*wn_model];
B = [0; 1];
C = eye(2);
D = [0;0];

% *** Liapunov steps: pick Q and solve for P 
% Hint: use symbolic representation and the solve function

% *** use P in the equation for M and solve for u, the control law

%% run the continuous simulink model
sim('student_model_ref');

%% Retrieve Desired and Actual States (You will need to modify the following code)
% xd = 
% x = 
% u = 
% times = 

%% Get variables to run simulink animation file (You will need to modify the following code)
% State data for running animation 
% x2 = ; (the theta*L state)
% deflAngle = [times, x2/L];
% trolleyPos = zeros(length(times),1);
% for aa = 2:length(times)
%     trolleyPos(aa) = trapz(times(aa-1:aa),u(aa-1:aa)) + trolleyPos(aa-1);
% end
% xt = [times, trolleyPos];

%% Run simulink animation file
sim('student_model_animation');