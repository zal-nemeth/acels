//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: xswap.cpp
//
// MATLAB Coder version            : 5.2
// C/C++ source code generated on  : 10-Apr-2021 12:17:27
//

// Include Files
#include "xswap.h"
#include "rt_nonfinite.h"

// Function Definitions
//
// Arguments    : double x[45]
//                int ix0
//                int iy0
// Return Type  : void
//
namespace coder {
namespace internal {
namespace blas {
void b_xswap(double x[45], int ix0, int iy0)
{
  for (int k{0}; k < 9; k++) {
    double temp;
    int i;
    int temp_tmp;
    temp_tmp = (ix0 + k) - 1;
    temp = x[temp_tmp];
    i = (iy0 + k) - 1;
    x[temp_tmp] = x[i];
    x[i] = temp;
  }
}

//
// Arguments    : double x[25]
//                int ix0
//                int iy0
// Return Type  : void
//
void xswap(double x[25], int ix0, int iy0)
{
  for (int k{0}; k < 5; k++) {
    double temp;
    int i;
    int temp_tmp;
    temp_tmp = (ix0 + k) - 1;
    temp = x[temp_tmp];
    i = (iy0 + k) - 1;
    x[temp_tmp] = x[i];
    x[i] = temp;
  }
}

} // namespace blas
} // namespace internal
} // namespace coder

//
// File trailer for xswap.cpp
//
// [EOF]
//
