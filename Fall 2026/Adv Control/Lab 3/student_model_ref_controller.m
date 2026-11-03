function u=student_model_ref_controller(xd1, xd2, v, x1, x2)
global Ap A P;
u = (A(2,1) - Ap(2,1))*x1 + (A(2,2) - Ap(2,2))*x2 + v;
return;