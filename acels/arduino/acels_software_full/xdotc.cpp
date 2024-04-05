//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: xdotc.cpp
//
// MATLAB Coder version            : 5.6
// C/C++ source code generated on  : 31-Mar-2024 11:00:20
//

// Include Files
#include "xdotc.h"
#include "rt_nonfinite.h"

// Function Definitions
//
// Arguments    : int n
//                const double x[25]
//                int ix0
//                const double y[25]
//                int iy0
// Return Type  : double
//
namespace coder {
namespace internal {
namespace blas {
double b_xdotc(int n, const double x[25], int ix0, const double y[25], int iy0)
{
  double d;
  int i;
  d = 0.0;
  i = static_cast<unsigned char>(n);
  for (int k{0}; k < i; k++) {
    d += x[(ix0 + k) - 1] * y[(iy0 + k) - 1];
  }
  return d;
}

//
// Arguments    : int n
//                const double x[45]
//                int ix0
//                const double y[45]
//                int iy0
// Return Type  : double
//
double xdotc(int n, const double x[45], int ix0, const double y[45], int iy0)
{
  double d;
  int i;
  d = 0.0;
  i = static_cast<unsigned char>(n);
  for (int k{0}; k < i; k++) {
    d += x[(ix0 + k) - 1] * y[(iy0 + k) - 1];
  }
  return d;
}

} // namespace blas
} // namespace internal
} // namespace coder

//
// File trailer for xdotc.cpp
//
// [EOF]
//
