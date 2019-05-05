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

	Matrix mat_mul(const Matrix& A, const Matrix& B);

	void mat_mul(const Matrix& A, const Matrix& B, Matrix& C,
		int , int , int, int);

	void qr(const Matrix& A, Matrix& Q, Matrix& R);

	float trace(const Matrix& A);

	Matrix trans(const Matrix& mat);

	void sort_index(Vector& v, Vector& idx);

	void sort_eigens(Matrix& V, Matrix& D);

	void eig(const Matrix& A, Matrix& V, Matrix& D);

	void mean_col(const Matrix& A, Matrix& m);

	void mat_sub(const Matrix& A, const Matrix& B, Matrix& C);

	void mat_add(const Matrix& A, const Matrix& B, Matrix& C);

	void pca(const Matrix& A, Matrix& A_red, const int k_col);

}

#endif