function [Q, R] = my_qr(A)
	[m, n] = size(A);
	Q = eye(m);
	R = A;

	for j = 1:n
		u = R(j:end,j);
		normx = norm(u);
		
		if u(1) < 0
			normx = -normx;
		end		

		u(1) = u(1) + normx;
		tau = u(1) / normx;
		u = u/u(1);
		H = eye(size(u,1)) - tau*u*u';
		R(j:end,j:end) = R(j:end,j:end) - tau * (u * u') * R(j:end,j:end);
		Q(:,j:end) = Q(:,j:end) - Q(:,j:end)*(tau * u * u');
	end

	Q*R;