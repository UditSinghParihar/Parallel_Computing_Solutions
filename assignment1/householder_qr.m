% 3.a, 3.b, 3.e

row = 3; col = 2;
A = rand(row, col)
b = rand(row, 1)

[m,n] = size(A);
Q = eye(m);
R = A; 

for j = 1:n
	normx = norm(R(j:end,j));
	s = sign(R(j,j));
	
	w = R(j:end,j);
	w(1) = R(j,j) + s*normx;
	w = w/w(1);

	tau = (2*s) / (w' * w);
	R(j:end,:) = R(j:end,:) - tau * (w * w') * R(j:end,:);
	Q(:,j:end) = Q(:,j:end) - Q(:,j:end)*(tau * w * w');
end

Q = Q(:, 1:col);
R = R(1:col, :);
b_new = Q' * b;
x = zeros(size(A,2), 1);

row = size(R,1);
x(row) = b_new(row)/R(row, row);
for i = row-1 : -1 : 1
	total = b_new(i);
	for j = row : -1 : i+1
		total = total - R(i, j)*x(j);
	end
	x(i) = total/R(i,i);
end
x

disp('Error: ');
norm((A * x) - b)