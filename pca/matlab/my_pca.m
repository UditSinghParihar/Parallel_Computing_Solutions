function [] = my_pca(img_name)
	img = imread(img_name);
	img = im2double(rgb2gray(img));
	imshow(img);
	drawnow;

	img = imresize(img, 0.8);
	
	m = mean(img);
	[row, col] = size(img);
	m = repmat(m, row, 1);
	img = img - m;
	
	c = img'*img;
	[v,d] = eig(c);
	[v,d] =  sortem(v,d);

	[sz, sz] = size(v);
	% v = v(:, 1:sz/2);
	v = v(:,1:10);
	size(v)

	% red_img = v*v'*img';
	% red_img = red_img' + m;
	red_img = img*v*v' + m;


	imshow(red_img);
	drawnow;

function [P2,D2]=sortem(P,D)
	D2=diag(sort(diag(D),'descend'));
	[c, ind]=sort(diag(D),'descend');
	P2=P(:,ind);