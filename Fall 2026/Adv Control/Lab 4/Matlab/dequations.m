function qdot = dequations(T,q,u,L,mp,mc,b,Mode)

%Define some parameters that we can use in our differential equations.

mt = mp+mc;
g = 9.81;
t = q(1);
w = q(2);
x = q(3);
v = q(4);
st = sin(t);
ct = cos(t);

switch Mode
    
    case 'Force'
        F = u;
        qdot(1,1) = w;
        qdot(2,1) = g/L * st -b/(mp*L)*w + ct/(L*(mc+mp*st^2))*(F - mp*L*w^2*st+mp*g*st*ct);
        qdot(3,1) = v;
        qdot(4,1) = F/(mc+mp*st^2) - mp*L*w^2*st/(mc+mp*st^2) + mp*g*st*ct/(mc+mp*st^2);     

    otherwise %these equations are actually for acceleration control, but the mode
        %for the pendulum class says velocity, and we just differentiate.
        A = u;
        qdot(1,1) = w;
        qdot(2,1) = g/L * st - b/(mp*L)*w + ct/L * A;
        qdot(3,1) = v;
        qdot(4,1) = A; 
end


