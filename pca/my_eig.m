function [V,D] = my_eig(A)
	eps = 1e-15;
	is_converged = 0;
	sum_prev = -10;
	change =0;
	steps=0;

	V = eye(size(A));

	while ~is_converged
		[q, r] = my_qr(A);
		A = r*q;
		V = V*q;

		sum = trace(A);
		change = sum - sum_prev;
				
		if abs(change) < eps
			is_converged = 1;
		end

		sum_prev = sum;
		steps = steps + 1;
	end

	D = A;
	change;
	steps
