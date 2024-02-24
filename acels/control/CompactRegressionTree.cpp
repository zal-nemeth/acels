//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: CompactRegressionTree.cpp
//
// MATLAB Coder version            : 5.2
// C/C++ source code generated on  : 10-Apr-2021 12:17:27
//

// Include Files
#include "CompactRegressionTree.h"
#include "rt_nonfinite.h"
#include "table.h"
#include "coder_bounded_array.h"
#include <cmath>

// Function Definitions
//
// Arguments    : const table *Xin
// Return Type  : double
//
namespace coder {
namespace classreg {
namespace learning {
namespace regr {
double CompactRegressionTree::predict(const table *Xin) const
{
  double X[2];
  int m;
  bool exitg1;
  X[0] = Xin->data[0];
  X[1] = Xin->data[1];
  m = 0;
  exitg1 = false;
  while (!(exitg1 || (this->PruneList.data[m] <= 0.0))) {
    double d;
    d = X[static_cast<int>(this->CutPredictorIndex[m]) - 1];
    if (std::isnan(d) || this->NanCutPoints[m]) {
      exitg1 = true;
    } else if (d < this->CutPoint[m]) {
      m = static_cast<int>(this->Children[m << 1]) - 1;
    } else {
      m = static_cast<int>(this->Children[(m << 1) + 1]) - 1;
    }
  }
  return this->NodeMean[m];
}

//
// Arguments    : const table *Xin
// Return Type  : double
//
double b_CompactRegressionTree::predict(const table *Xin) const
{
  double X[2];
  int m;
  bool exitg1;
  X[0] = Xin->data[0];
  X[1] = Xin->data[1];
  m = 0;
  exitg1 = false;
  while (!(exitg1 || (this->PruneList.data[m] <= 0.0))) {
    double d;
    d = X[static_cast<int>(this->CutPredictorIndex[m]) - 1];
    if (std::isnan(d) || this->NanCutPoints[m]) {
      exitg1 = true;
    } else if (d < this->CutPoint[m]) {
      m = static_cast<int>(this->Children[m << 1]) - 1;
    } else {
      m = static_cast<int>(this->Children[(m << 1) + 1]) - 1;
    }
  }
  return this->NodeMean[m];
}

//
// Arguments    : const table *Xin
// Return Type  : double
//
double c_CompactRegressionTree::predict(const table *Xin) const
{
  double X[2];
  int m;
  bool exitg1;
  X[0] = Xin->data[0];
  X[1] = Xin->data[1];
  m = 0;
  exitg1 = false;
  while (!(exitg1 || (this->PruneList.data[m] <= 0.0))) {
    double d;
    d = X[static_cast<int>(this->CutPredictorIndex[m]) - 1];
    if (std::isnan(d) || this->NanCutPoints[m]) {
      exitg1 = true;
    } else if (d < this->CutPoint[m]) {
      m = static_cast<int>(this->Children[m << 1]) - 1;
    } else {
      m = static_cast<int>(this->Children[(m << 1) + 1]) - 1;
    }
  }
  return this->NodeMean[m];
}

} // namespace regr
} // namespace learning
} // namespace classreg
} // namespace coder

//
// File trailer for CompactRegressionTree.cpp
//
// [EOF]
//
