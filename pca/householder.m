function [] = householder(x)	
	u = x;
	normx = norm(u)
	
	if u(1) < 0
		normx = -normx;
	end		

	u(1) = u(1) + normx;
	tau = u(1) / normx;
	u = u/u(1);
	
	I = eye(size(u,1));
	P =  I - tau*u*u'

	P*x;
	