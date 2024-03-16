//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: varNamesDim.h
//
// MATLAB Coder version            : 5.2
// C/C++ source code generated on  : 10-Apr-2021 12:17:27
//

#ifndef VARNAMESDIM_H
#define VARNAMESDIM_H

// Include Files
#include "rtwtypes.h"
#include "coder_bounded_array.h"
#include <cstddef>
#include <cstdlib>

// Type Definitions
namespace coder {
namespace matlab {
namespace internal {
namespace coder {
namespace tabular {
enum Continuity
{
  unset = 0, // Default value
  continuous,
  step,
  event
};

}
} // namespace coder
} // namespace internal
} // namespace matlab
} // namespace coder
struct cell_wrap_2 {
  coder::empty_bounded_array<char, 2U> f1;
};

namespace coder {
namespace matlab {
namespace internal {
namespace coder {
namespace tabular {
namespace private_ {
class varNamesDim {
public:
  cell_wrap_2 descrs[2];
  cell_wrap_2 units[2];
  Continuity continuity[2];
  bool hasDescrs;
  bool hasUnits;
  bool hasContinuity;
};

} // namespace private_
} // namespace tabular
} // namespace coder
} // namespace internal
} // namespace matlab
} // namespace coder

#endif
//
// File trailer for varNamesDim.h
//
// [EOF]
//
