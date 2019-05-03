#ifndef LINEAR_ALGEBRA_H
#define LINEAR_ALGEBRA_H

#include <vector>

namespace la{

	typedef std::vector<float> Vector;
	typedef std::vector<Vector> Matrix;

	void fill_matrix(Matrix& mat, int range);

	void print_matrix(const Matrix& mat);

	void fill_vector(Vector& col, int range);

	void print_vector(const Vector& vec);

	void check_householder(const Matrix& P, const Vector& x);

	void householder(Vector u, Matrix& P);

	Matrix matrix_mul(const Matrix& A, const Matrix& B);

	void matrix_mul(const Matrix& A, const Matrix& B, Matrix& C,
		int , int , int, int);

	void qr(const Matrix& A, Matrix& Q, Matrix& R);
}

#endif