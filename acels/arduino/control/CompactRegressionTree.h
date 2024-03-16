//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: CompactRegressionTree.h
//
// MATLAB Coder version            : 5.2
// C/C++ source code generated on  : 10-Apr-2021 12:17:27
//

#ifndef COMPACTREGRESSIONTREE_H
#define COMPACTREGRESSIONTREE_H

// Include Files
#include "rtwtypes.h"
#include "coder_bounded_array.h"
#include <cstddef>
#include <cstdlib>

// Type Declarations
namespace coder {
class table;

}

// Type Definitions
namespace coder {
namespace classreg {
namespace learning {
namespace coderutils {
enum Transform
{
  Logit = 0, // Default value
  Doublelogit,
  Invlogit,
  Ismax,
  Sign,
  Symmetric,
  Symmetricismax,
  Symmetriclogit,
  Identity
};

}
namespace regr {
class CompactRegressionTree {
public:
  double predict(const table *Xin) const;
  double CutPredictorIndex[91];
  double Children[182];
  double CutPoint[91];
  bounded_array<double, 91U, 1U> PruneList;
  bool NanCutPoints[91];
  bool InfCutPoints[91];
  coderutils::Transform ResponseTransform;
  double NodeMean[91];
};

class b_CompactRegressionTree {
public:
  double predict(const table *Xin) const;
  double CutPredictorIndex[101];
  double Children[202];
  double CutPoint[101];
  bounded_array<double, 101U, 1U> PruneList;
  bool NanCutPoints[101];
  bool InfCutPoints[101];
  coderutils::Transform ResponseTransform;
  double NodeMean[101];
};

class c_CompactRegressionTree {
public:
  double predict(const table *Xin) const;
  double CutPredictorIndex[119];
  double Children[238];
  double CutPoint[119];
  bounded_array<double, 119U, 1U> PruneList;
  bool NanCutPoints[119];
  bool InfCutPoints[119];
  coderutils::Transform ResponseTransform;
  double NodeMean[119];
};

} // namespace regr
} // namespace learning
} // namespace classreg
} // namespace coder

#endif
//
// File trailer for CompactRegressionTree.h
//
// [EOF]
//
