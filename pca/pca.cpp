#include <iostream>
#include "linear_algebra.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace la;
using namespace cv;

void disp(const Mat& img){
	namedWindow("opencv_viewer", WINDOW_AUTOSIZE);
	imshow("opencv_viewer", img);
	waitKey(0);
}

void Mat2Matrix(const Mat& gray, Matrix& mat){
	const int rows = mat.size(), cols = mat[0].size();
	
	for(int i=0; i<rows; ++i)
		for(int j=0; j<cols; ++j)
			mat[i][j] = float(gray.at<uchar>(i, j));

	for(int i=0; i<cols; ++i)
		cout << mat[120][i] << " ";
	cout << "\n--\n";
}

void Matrix2Mat(){}

void compress(const string& name){
	Mat color = imread(name, IMREAD_COLOR);
	const int scale = 3.5;
	// resize(color, color, Size(color.cols/scale, color.rows/scale));
	fprintf(stdout, "rows: %d\tcols: %d\n--\n", color.rows, color.cols);
	
	Mat gray;
	cvtColor(color, gray, COLOR_BGR2GRAY);
	// disp(color);
	disp(gray);

	// cout << gray.type() << endl << color.type() << endl;

	// for(int i=0; i<gray.rows; ++i)
	// 	for(int j=0; j<gray.cols; ++j)
	// 		if(i==j)
	// 			gray.at<uchar>(i, j) = 255;

	// for(int i=0; i<gray.cols; ++i)
	// 	gray.at<uchar>(120, i) = 255;

	// disp(gray);

	for(int i=0; i<gray.cols; ++i)
		cout << float(gray.at<uchar>(120, i)) << " ";
	cout << "\n--\n";

	Matrix mat(gray.rows, Vector(gray.cols));
	Mat2Matrix(gray, mat);

	// Matrix mat_red(gray.rows, Vector(gray.cols));
	// const int k_col = 10;
	// pca(mat, mat_red, k_col);

	// for(int i=0; i<gray.cols; ++i)
	// 	cout << mat_red[120][i] << " ";
	// cout << "\n--\n";
}

int main(int argc, char const *argv[]){
	if(argc != 2){
		fprintf(stdout, "Usage: %s image.jpg\n", argv[0]);
		return 1;
	}

	// const int rows_a=20, cols_a=8, range=10;
	// Matrix A(rows_a, Vector(cols_a));
	// fill_matrix(A, range);
	// print_matrix(A);

	// Matrix A_red(rows_a, Vector(cols_a));
	// const int k_col = 4;
	// pca(A, A_red, k_col);
	// print_matrix(A_red);

	compress(argv[1]);

	return 0;
}