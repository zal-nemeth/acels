//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: inpolygon.cpp
//
// MATLAB Coder version            : 5.6
// C/C++ source code generated on  : 31-Mar-2024 11:00:20
//

// Include Files
#include "inpolygon.h"
#include "rt_nonfinite.h"
#include <cmath>

// Function Definitions
//
// Arguments    : double x1
//                double b_y1
//                double x2
//                double y2
//                signed char quad1
//                signed char quad2
//                double scale
//                bool &onj
// Return Type  : signed char
//
namespace coder {
signed char contrib(double x1, double b_y1, double x2, double y2,
                    signed char quad1, signed char quad2, double scale,
                    bool &onj)
{
  double cp;
  signed char diffQuad;
  onj = false;
  diffQuad = static_cast<signed char>(quad2 - quad1);
  cp = x1 * y2 - x2 * b_y1;
  if (std::abs(cp) < scale) {
    onj = (x1 * x2 + b_y1 * y2 <= 0.0);
    if ((diffQuad == 2) || (diffQuad == -2)) {
      diffQuad = 0;
    } else if (diffQuad == -3) {
      diffQuad = 1;
    } else if (diffQuad == 3) {
      diffQuad = -1;
    }
  } else if (cp < 0.0) {
    if (diffQuad == 2) {
      diffQuad = -2;
    } else if (diffQuad == -3) {
      diffQuad = 1;
    } else if (diffQuad == 3) {
      diffQuad = -1;
    }
  } else if (diffQuad == -2) {
    diffQuad = 2;
  } else if (diffQuad == -3) {
    diffQuad = 1;
  } else if (diffQuad == 3) {
    diffQuad = -1;
  }
  return diffQuad;
}

} // namespace coder

//
// File trailer for inpolygon.cpp
//
// [EOF]
//
