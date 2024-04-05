//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: xdotc.h
//
// MATLAB Coder version            : 5.6
// C/C++ source code generated on  : 31-Mar-2024 11:00:20
//

#ifndef XDOTC_H
#define XDOTC_H

// Include Files
#include "rtwtypes.h"
#include <cstddef>
#include <cstdlib>

// Function Declarations
namespace coder {
namespace internal {
namespace blas {
double b_xdotc(int n, const double x[25], int ix0, const double y[25], int iy0);

double xdotc(int n, const double x[45], int ix0, const double y[45], int iy0);

} // namespace blas
} // namespace internal
} // namespace coder

#endif
//
// File trailer for xdotc.h
//
// [EOF]
//
