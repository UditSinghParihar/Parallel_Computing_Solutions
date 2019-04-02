function [] = my_pca(img_name)
	img = imread(img_name);
	img = im2double(rgb2gray(img));
	
	M = mean(img, 2);

	for i = size(M,2)
		img(:,i) = img(:,i) - M;
	end
	
	c = img'*img;
	[v,d] = eig(c);
	[v,d] =  sortem(v,d);

	[m, m] = size(v);
	v = v(:, 1:m/2);

	red_img = img*v;

	imshow(red_img);
	drawnow;

function [P2,D2]=sortem(P,D)
	D2=diag(sort(diag(D),'descend'));
	[c, ind]=sort(diag(D),'descend');
	P2=P(:,ind); 
