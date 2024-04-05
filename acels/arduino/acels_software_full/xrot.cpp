//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: xrot.cpp
//
// MATLAB Coder version            : 5.6
// C/C++ source code generated on  : 31-Mar-2024 11:00:20
//

// Include Files
#include "xrot.h"
#include "rt_nonfinite.h"

// Function Definitions
//
// Arguments    : double x[45]
//                int ix0
//                int iy0
//                double c
//                double s
// Return Type  : void
//
namespace coder {
namespace internal {
namespace blas {
void b_xrot(double x[45], int ix0, int iy0, double c, double s)
{
  for (int k{0}; k < 9; k++) {
    double b_temp_tmp;
    double temp_tmp;
    int b_temp_tmp_tmp;
    int temp_tmp_tmp;
    temp_tmp_tmp = (iy0 + k) - 1;
    temp_tmp = x[temp_tmp_tmp];
    b_temp_tmp_tmp = (ix0 + k) - 1;
    b_temp_tmp = x[b_temp_tmp_tmp];
    x[temp_tmp_tmp] = c * temp_tmp - s * b_temp_tmp;
    x[b_temp_tmp_tmp] = c * b_temp_tmp + s * temp_tmp;
  }
}

//
// Arguments    : double x[25]
//                int ix0
//                int iy0
//                double c
//                double s
// Return Type  : void
//
void xrot(double x[25], int ix0, int iy0, double c, double s)
{
  for (int k{0}; k < 5; k++) {
    double b_temp_tmp;
    double temp_tmp;
    int b_temp_tmp_tmp;
    int temp_tmp_tmp;
    temp_tmp_tmp = (iy0 + k) - 1;
    temp_tmp = x[temp_tmp_tmp];
    b_temp_tmp_tmp = (ix0 + k) - 1;
    b_temp_tmp = x[b_temp_tmp_tmp];
    x[temp_tmp_tmp] = c * temp_tmp - s * b_temp_tmp;
    x[b_temp_tmp_tmp] = c * b_temp_tmp + s * temp_tmp;
  }
}

} // namespace blas
} // namespace internal
} // namespace coder

//
// File trailer for xrot.cpp
//
// [EOF]
//
