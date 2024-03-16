//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: xaxpy.h
//
// MATLAB Coder version            : 5.2
// C/C++ source code generated on  : 10-Apr-2021 12:17:27
//

#ifndef XAXPY_H
#define XAXPY_H

// Include Files
#include "rtwtypes.h"
#include <cstddef>
#include <cstdlib>

// Function Declarations
namespace coder {
namespace internal {
namespace blas {
void b_xaxpy(int n, double a, const double x[9], int ix0, double y[45],
             int iy0);

void b_xaxpy(int n, double a, int ix0, double y[25], int iy0);

void xaxpy(int n, double a, const double x[45], int ix0, double y[9], int iy0);

void xaxpy(int n, double a, int ix0, double y[45], int iy0);

} // namespace blas
} // namespace internal
} // namespace coder

#endif
//
// File trailer for xaxpy.h
//
// [EOF]
//
