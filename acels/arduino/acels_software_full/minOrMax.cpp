//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: minOrMax.cpp
//
// MATLAB Coder version            : 5.6
// C/C++ source code generated on  : 31-Mar-2024 11:00:20
//

// Include Files
#include "minOrMax.h"
#include "rt_nonfinite.h"
#include <cmath>

// Function Definitions
//
// Arguments    : const double x[336]
//                int &idx
// Return Type  : double
//
namespace coder {
namespace internal {
double minimum(const double x[336], int &idx)
{
  double ex;
  int k;
  if (!std::isnan(x[0])) {
    idx = 1;
  } else {
    bool exitg1;
    idx = 0;
    k = 2;
    exitg1 = false;
    while ((!exitg1) && (k < 337)) {
      if (!std::isnan(x[k - 1])) {
        idx = k;
        exitg1 = true;
      } else {
        k++;
      }
    }
  }
  if (idx == 0) {
    ex = x[0];
    idx = 1;
  } else {
    int i;
    ex = x[idx - 1];
    i = idx + 1;
    for (k = i; k < 337; k++) {
      double d;
      d = x[k - 1];
      if (ex > d) {
        ex = d;
        idx = k;
      }
    }
  }
  return ex;
}

} // namespace internal
} // namespace coder

//
// File trailer for minOrMax.cpp
//
// [EOF]
//
