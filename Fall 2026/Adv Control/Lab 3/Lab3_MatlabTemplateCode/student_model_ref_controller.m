function u=student_model_ref_controller(xd1, xd2, v, x1, x2)

% declaring these variables global so we can see them here
global Ap  A  P;

% errors
e1 = xd1 - x1;
e2 = xd2 - x2;

% insert the control law here:
% u should be in m/s
u=1;

return;