function [] = matrix_pca(A)
	% A = imread(img_name);
	% A = im2double(rgb2gray(A));
	% A = imresize(A, 0.8);
	
	m = mean(A);
	[row, col] = size(A);
	m = repmat(m, row, 1);
	A = A - m;
	
	c = A'*A;
	[v,d] = eig(c);
	[v,d] =  sortem(v,d);

	[sz, sz] = size(v);
	% v = v(:, 1:sz/2);
	v = v(:,1:2);
	size(v);

	red_A = A*v*v' + m

	% imshow(red_A);
	% drawnow;

function [P2,D2]=sortem(P,D)
	D2=diag(sort(diag(D),'descend'));
	[c, ind]=sort(diag(D),'descend');
	P2=P(:,ind);