A = [4, 7, 8; 6, 4, 6; 7, 3, 10; 2 3 8; 1 10 4; 7 1 7];

%U = A;
%L = eye(3);
%m=3;
%for k = 1:(m-1)
%	for j = (k+1) : m
%		L(j,k) = U(j,k)/U(k,k);
%		U(j,k:m) = U(j,k:m) - L(j,k)*U(k,k:m);
%	end
%end
%
%L
%U
%L*U

m=3;
for k = 1 : m-1
	A(k+1:m, k) = A(k+1:m, k)/A(k,k);
	for i = k+1 : m
		for j = k+1 : m
			A(i, j) = A(i, j) - A(i, k)*A(k, j);
		end
	end
end

A