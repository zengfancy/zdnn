#include "match_func.h"

// C = alpha * A * B + beta * C
// C: m * n
// A: m * k
// B: k * n
void dgemm(double* A, bool transA,
    double* B, bool transB,
    double alpha,
    double* C,
    double beta,
    int m, int n, int k) {
  cublasDgemm(handle, transA, transB,
      m, n, k,
      &alpha,
      A, m,
      B, k,
      &beta,
      C, m);
}


// matrix-vector multiplication, y = alpha * A * x + beta * y
// A: m * n
void dgemv(double* A, bool trans,
    double* x,
    double alpha,
    double* y,
    double beta,
    int m, int n) {
  cublasDgemv(handle, trans,
      m, n,
      &alpha,
      A, m, // leading dimension of A
      x, 1/*incx*/,
      &beta,
      y, 1/*incy*/);

}


