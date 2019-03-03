x = [4; 7];
I = eye(size(x, 1));
tau = norm(x);

v = x;
s = sign(x(1));
v(1) = v(1) + s*tau;
v = v/v(1)

b = (2*s) / (v' * v)
P = I - b*v*v'
P*x
norm(x)