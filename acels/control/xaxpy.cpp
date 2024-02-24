//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: xaxpy.cpp
//
// MATLAB Coder version            : 5.2
// C/C++ source code generated on  : 10-Apr-2021 12:17:27
//

// Include Files
#include "xaxpy.h"
#include "rt_nonfinite.h"

// Function Definitions
//
// Arguments    : int n
//                double a
//                const double x[9]
//                int ix0
//                double y[45]
//                int iy0
// Return Type  : void
//
namespace coder {
namespace internal {
namespace blas {
void b_xaxpy(int n, double a, const double x[9], int ix0, double y[45], int iy0)
{
  if (!(a == 0.0)) {
    int i;
    i = n - 1;
    for (int k{0}; k <= i; k++) {
      int i1;
      i1 = (iy0 + k) - 1;
      y[i1] += a * x[(ix0 + k) - 1];
    }
  }
}

//
// Arguments    : int n
//                double a
//                int ix0
//                double y[25]
//                int iy0
// Return Type  : void
//
void b_xaxpy(int n, double a, int ix0, double y[25], int iy0)
{
  if (!(a == 0.0)) {
    int i;
    i = n - 1;
    for (int k{0}; k <= i; k++) {
      int i1;
      i1 = (iy0 + k) - 1;
      y[i1] += a * y[(ix0 + k) - 1];
    }
  }
}

//
// Arguments    : int n
//                double a
//                const double x[45]
//                int ix0
//                double y[9]
//                int iy0
// Return Type  : void
//
void xaxpy(int n, double a, const double x[45], int ix0, double y[9], int iy0)
{
  if (!(a == 0.0)) {
    int i;
    i = n - 1;
    for (int k{0}; k <= i; k++) {
      int i1;
      i1 = (iy0 + k) - 1;
      y[i1] += a * x[(ix0 + k) - 1];
    }
  }
}

//
// Arguments    : int n
//                double a
//                int ix0
//                double y[45]
//                int iy0
// Return Type  : void
//
void xaxpy(int n, double a, int ix0, double y[45], int iy0)
{
  if (!(a == 0.0)) {
    int i;
    i = n - 1;
    for (int k{0}; k <= i; k++) {
      int i1;
      i1 = (iy0 + k) - 1;
      y[i1] += a * y[(ix0 + k) - 1];
    }
  }
}

} // namespace blas
} // namespace internal
} // namespace coder

//
// File trailer for xaxpy.cpp
//
// [EOF]
//
