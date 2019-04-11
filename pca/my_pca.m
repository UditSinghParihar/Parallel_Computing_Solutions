function [] = my_pca(img_name)
	img = imread(img_name);
	img = im2double(rgb2gray(img));
	img = imresize(img, 0.8);
	
	m = mean(img);
	[row, col] = size(img);
	m = repmat(m, row, 1);
	img = img - m;
	
	c = img'*img;
	[v,d] = my_eig(c);
	[v,d] =  sortem(v,d);

	[r, r] = size(v);
	% v = v(:, 1:r/2);
	v = v(:,1:10);
	size(v)

	red_img = v*v'*img';
	red_img = red_img' + m;

	imshow(red_img);
	drawnow;

function [P2,D2]=sortem(P,D)
	D2=diag(sort(diag(D),'descend'));
	[c, ind]=sort(diag(D),'descend');
	P2=P(:,ind); 
