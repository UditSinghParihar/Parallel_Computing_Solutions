function [V,D] = my_eig(A)
	eps = 1e-15;
	is_converged = 0;
	sum_prev = -10;
	change =0;
	steps=0;

	D = A;
	V = eye(size(D));

	while ~is_converged
		[q, r] = my_qr(D);
		D = r*q;
		V = V*q;

		sum = trace(D);
		change = sum - sum_prev;
				
		if abs(change) < eps
			is_converged = 1;
		end

		sum_prev = sum;
		steps = steps + 1;
	end

	[V,D] =  sortem(V,D);

	change;
	steps

function [P2,D2]=sortem(P,D)
	D2=diag(sort(diag(D),'descend'));
	[c, ind]=sort(diag(D),'descend');
	P2=P(:,ind);