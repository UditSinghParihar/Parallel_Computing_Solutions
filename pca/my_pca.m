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
	
	[sz, sz] = size(v);
	% v = v(:, 1:sz/2);
	v = v(:,1:10);
	size(v)

	% red_img = v*v'*img';
	% red_img = red_img' + m;
	red_img = img*v*v' + m;


	imshow(red_img);
	drawnow;
