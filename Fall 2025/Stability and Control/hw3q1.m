k = 1; m0 = 5; mi = 20;
a = k/m0; b = k/mi;

A = [ 0  1  0  0  0  0;
     -2*a 0  a  0  0  0;
      0  0  0  1  0  0;
      b  0 -2*b 0  b  0;
      0  0  0  0  0  1;
      0  0  a  0 -2*a 0 ];

[V,D] = eig(A);
lambda = diag(D);
disp(lambda)