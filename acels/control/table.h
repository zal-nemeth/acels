//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: table.h
//
// MATLAB Coder version            : 5.2
// C/C++ source code generated on  : 10-Apr-2021 12:17:27
//

#ifndef TABLE_H
#define TABLE_H

// Include Files
#include "metaDim.h"
#include "rowNamesDim.h"
#include "rtwtypes.h"
#include "varNamesDim.h"
#include "coder_bounded_array.h"
#include <cstddef>
#include <cstdlib>

// Type Definitions
struct struct_T {
  coder::empty_bounded_array<char, 2U> Description;
};

namespace coder {
class table {
public:
  void parenReference(table *b) const;
  matlab::internal::coder::tabular::private_::varNamesDim varDim;
  double data[2];
  struct_T arrayProps;

protected:
  matlab::internal::coder::tabular::private_::metaDim b_metaDim;
  matlab::internal::coder::tabular::private_::rowNamesDim rowDim;
};

} // namespace coder

#endif
//
// File trailer for table.h
//
// [EOF]
//
