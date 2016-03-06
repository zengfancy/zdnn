#ifndef __MATH_FUNC__
#define __MATH_FUNC__

template <class T>
T sigmoid(T x) {
  return 1 / (1 + exp(-x));
}

template <class T>
T d_sigmoid(T x) {
  exp_minus_x = exp(-x);
  return exp_minus_x / ((1 + exp_minus_x) * (1 + exp_minus_x));
}

// matrix-matrix multiplication
// C = alpha * A * B + beta * C
void dgemm(double* A, bool trans,
    double* B, bool trans,
    double alpha,
    double* C, bool trans,
    double beta,
    int m, int n, int k);


// matrix-vector multiplication, y = alpha * A * x + beta * y
void dgemv(double* A, bool trans,
    double* x,
    double alpha,
    double* y,
    double beta,
    int m, int n);


#endif
