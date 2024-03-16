//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: table.cpp
//
// MATLAB Coder version            : 5.2
// C/C++ source code generated on  : 10-Apr-2021 12:17:27
//

// Include Files
#include "table.h"
#include "matrix_transform_v1_data.h"
#include "metaDim.h"
#include "rt_nonfinite.h"
#include "varNamesDim.h"
#include "coder_bounded_array.h"

// Function Definitions
//
// Arguments    : table *b
// Return Type  : void
//
namespace coder {
void table::parenReference(table *b) const
{
  b->varDim.hasUnits = false;
  b->varDim.units[0].f1.size[0] = 1;
  b->varDim.units[0].f1.size[1] = 0;
  b->varDim.units[1].f1.size[0] = 1;
  b->varDim.units[1].f1.size[1] = 0;
  b->varDim.hasDescrs = false;
  b->varDim.descrs[0].f1.size[0] = 1;
  b->varDim.descrs[0].f1.size[1] = 0;
  b->varDim.descrs[1].f1.size[0] = 1;
  b->varDim.descrs[1].f1.size[1] = 0;
  b->varDim.hasContinuity = false;
  b->varDim.continuity[0] = matlab::internal::coder::tabular::unset;
  b->varDim.continuity[1] = matlab::internal::coder::tabular::unset;
  b->data[0] = this->data[0];
  b->data[1] = this->data[1];
  b->b_metaDim = this->b_metaDim;
  b->arrayProps = this->arrayProps;
}

} // namespace coder

//
// File trailer for table.cpp
//
// [EOF]
//
