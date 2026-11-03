function source
clear all
clear classes
clc

% Keyboard input uses MATLAB callbacks; click the pendulum figure to use keys.

%CONSTANTS
L  = 2.009;               %Length
mc = 3;                 %Mass of Cart
mp = 0.4781;                 %Mass of Payload
b  = 0.0513;               %Damping factor
g  = 9.81;              %Gravity
Mode = 'Acceleration';         %Velocity or Force or Acceleration
min_cycle = 0.05;       %Minimum cycle time

%PREPARE THE FIGURE WINDOW & GET PENDULUM OBJECT
fig1 = figure(1); 
fig1.Position = [0,0,900,1536];
clf;
kb = fig1;
setappdata(kb,'LabKeys',struct('LEFT',false,'RIGHT',false,'ESC',false));
set(kb,'WindowKeyPressFcn',{@KEY_STATE,true,kb},'WindowKeyReleaseFcn',{@KEY_STATE,false,kb});

%change initial states here
Pend = PENDULUM('InitialStates',[0 0 0 0], 'WorkspaceLength', 8,'Damping',b,...
    'MassCart', mc, 'MassPendulum', mp, 'PendulumLength',L,'Mode',Mode);

 
%PREPARE THE STREAMING FIGURE WINDOWS
fig2 = figure(2); clf
fig2.Position = [960,0,900,1536];

subplot(3,1,1);
Stream_Theta = STREAM_AXIS_DATA(gca);
ylabel('\theta')

subplot(3,1,2);
Stream_Pos = STREAM_AXIS_DATA(gca);
ylabel('Position')

subplot(3,1,3);
Stream_Vel = STREAM_AXIS_DATA(gca);
ylabel('Velocity')

%Keys work no matter which figure window has focus
set(fig2,'WindowKeyPressFcn',{@KEY_STATE,true,kb},'WindowKeyReleaseFcn',{@KEY_STATE,false,kb});
figure(fig1);


%BEGINNING OF SIMULATION LOOP
quit    = 0;
U_fbl   = 0;
del     = min_cycle;
int_past = 0;


while quit == 0
    
    %KEEP TRACK OF TIME
    now = GETTIME;
    
    %CHECK IF THE ESC, F1 KEY IS BEING PRESSED
    kbinput = read(kb); 
    if kbinput.ESC 
        quit = 1;
    end
    
    %GET VR FROM THE KEYBOARD
    U_manual = GET_MANUAL_COMMAND('LEFT','RIGHT',kb); % Use LEFT and RIGHT arrow keys
    
    %The States
    Q  = Pend.Q;

    t = Q(1);
    w = Q(2);
    x = Q(3);
    v = Q(4);
    st = sin(t);
    ct = cos(t);

    k0 = 16;
    k1 = 8;
    
    %Feedback Linearization   
    switch Mode
        case 'Force'   
                %U_fbl = ??
      
        case 'Velocity'
                U_fbl = 0;
            
         case 'Acceleration'
                U_fbl = L/ct * (-k0*t-k1*w-g*st/L + b*w/(mp*L));
    end
    
    %Full State Feedback (from LQR)
    switch Mode
        case 'Force'   
                K = [97.4286   27.1588   -1.0000   -3.4443];
                U_lqr = -K * [t; w; x; v];
                U_fbl = 0;
      
        case 'Velocity'
                K = [94.7511   71.2751   -5.0000];
                int = int_past*0.99 + t*del;
                int_past = int;
                U_lqr = -K * [int; t; x];

                
                %hint: this mode is sensitive to numerical error that manifests itself in the form of 'integral wind-up'. See if
                %you can understand why this is the case. To make this mode work reliably in this simulation environment, you need to 
                %mitigate the integral wind-up.
                
                %K       = [??  ??  ??];
                %z1      = z1*0.99 + Q(1)*del;               %0.99 mitigates integral wind-up in the z1 state caused by numerical errors.
                %z2      = Q(1);
                %z3      = Q(3);
                %Z       = [z1 z2 z3]';
                %U_lqr   = -K * Z;
            
         case 'Acceleration'
                K = [34.9649   10.3858   -1.0000   -2.2646];
                U_lqr = -K * [t; w; x; v];
                U_lqr = 0;
    end
  
    
    %Actuate the Pendulum 
    
    Pend.U = U_manual  + U_fbl + U_lqr;
         
    %Trace states
    Stream_Theta.stream(now, Q(1));
    Stream_Pos.stream(now, Q(3));
    Stream_Vel.stream(now, Q(4));
     
    %update screen
    drawnow
    
    %HANDLE ANY IDLE TIME
    finish = GETTIME;    
    while finish <= now + min_cycle;
        finish = GETTIME;
    end
    del = finish - now;
end


%--------------------------------------------------
function Uout = GET_MANUAL_COMMAND(xm, xp, kb)
umax 	= 1;
U       = 0; 
kbinput = read(kb);

if isfield(kbinput,xm) && isfield(kbinput,xp) % if using meta keys such as arrows
    if kbinput.(xp) % move right with specified meta key (e.g. RIGHT arrow)
        U = U + umax;
    end
    if kbinput.(xm) % move left with specified key (e.g. LEFT arrow)
        U = U - umax; 
    end
else % using regular keys
    if sum(kbinput.keys) ~= 0
        if kbinput.keys(xp)% move right with specified key (e.g. 'd')
            U = U + umax;
        end
        if kbinput.keys(xm) % move left with specified key (e.g. 'a')
            U = U - umax; 
        end
    end
end

Uout = U;

%--------------------------------------------------
function out = GETTIME
%time = GETTIME returns the current cpu clock time in milliseconds

tmp = clock;
out = 3600*tmp(4) + 60*tmp(5)+tmp(6);
     


function out = read(kb)
out = getappdata(kb,'LabKeys');

function KEY_STATE(~,event,pressed,kb)
s = getappdata(kb,'LabKeys');
switch event.Key
    case 'leftarrow',  s.LEFT = pressed;
    case 'rightarrow', s.RIGHT = pressed;
    case 'escape',     s.ESC = pressed;
end
setappdata(kb,'LabKeys',s);
