//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: svd.cpp
//
// MATLAB Coder version            : 5.6
// C/C++ source code generated on  : 31-Mar-2024 11:00:20
//

// Include Files
#include "svd.h"
#include "rt_nonfinite.h"
#include "xaxpy.h"
#include "xdotc.h"
#include "xnrm2.h"
#include "xrot.h"
#include "xrotg.h"
#include "xswap.h"
#include <algorithm>
#include <cmath>
#include <cstring>

// Function Definitions
//
// Arguments    : const double A[45]
//                double U[45]
//                double s[5]
//                double V[25]
// Return Type  : void
//
namespace coder {
namespace internal {
void svd(const double A[45], double U[45], double s[5], double V[25])
{
  double b_A[45];
  double Vf[25];
  double work[9];
  double b_s[5];
  double e[5];
  double nrm;
  double rt;
  double snorm;
  double sqds;
  int i;
  int ii;
  int jj;
  int m;
  int qp1;
  int qp1jj;
  int qq;
  std::copy(&A[0], &A[45], &b_A[0]);
  for (i = 0; i < 5; i++) {
    b_s[i] = 0.0;
    e[i] = 0.0;
  }
  std::memset(&work[0], 0, 9U * sizeof(double));
  std::memset(&U[0], 0, 45U * sizeof(double));
  std::memset(&Vf[0], 0, 25U * sizeof(double));
  for (int q{0}; q < 5; q++) {
    bool apply_transform;
    qp1 = q + 2;
    qp1jj = q + 9 * q;
    qq = qp1jj + 1;
    apply_transform = false;
    nrm = blas::xnrm2(9 - q, b_A, qp1jj + 1);
    if (nrm > 0.0) {
      apply_transform = true;
      if (b_A[qp1jj] < 0.0) {
        nrm = -nrm;
      }
      b_s[q] = nrm;
      if (std::abs(nrm) >= 1.0020841800044864E-292) {
        nrm = 1.0 / nrm;
        jj = (qp1jj - q) + 9;
        for (int k{qq}; k <= jj; k++) {
          b_A[k - 1] *= nrm;
        }
      } else {
        jj = (qp1jj - q) + 9;
        for (int k{qq}; k <= jj; k++) {
          b_A[k - 1] /= b_s[q];
        }
      }
      b_A[qp1jj]++;
      b_s[q] = -b_s[q];
    } else {
      b_s[q] = 0.0;
    }
    for (jj = qp1; jj < 6; jj++) {
      i = q + 9 * (jj - 1);
      if (apply_transform) {
        blas::xaxpy(
            9 - q,
            -(blas::xdotc(9 - q, b_A, qp1jj + 1, b_A, i + 1) / b_A[qp1jj]),
            qp1jj + 1, b_A, i + 1);
      }
      e[jj - 1] = b_A[i];
    }
    for (ii = q + 1; ii < 10; ii++) {
      i = (ii + 9 * q) - 1;
      U[i] = b_A[i];
    }
    if (q + 1 <= 3) {
      nrm = blas::b_xnrm2(4 - q, e, q + 2);
      if (nrm == 0.0) {
        e[q] = 0.0;
      } else {
        if (e[q + 1] < 0.0) {
          e[q] = -nrm;
        } else {
          e[q] = nrm;
        }
        nrm = e[q];
        if (std::abs(e[q]) >= 1.0020841800044864E-292) {
          nrm = 1.0 / e[q];
          for (int k{qp1}; k < 6; k++) {
            e[k - 1] *= nrm;
          }
        } else {
          for (int k{qp1}; k < 6; k++) {
            e[k - 1] /= nrm;
          }
        }
        e[q + 1]++;
        e[q] = -e[q];
        for (ii = qp1; ii < 10; ii++) {
          work[ii - 1] = 0.0;
        }
        for (jj = qp1; jj < 6; jj++) {
          blas::xaxpy(8 - q, e[jj - 1], b_A, (q + 9 * (jj - 1)) + 2, work,
                      q + 2);
        }
        for (jj = qp1; jj < 6; jj++) {
          blas::b_xaxpy(8 - q, -e[jj - 1] / e[q + 1], work, q + 2, b_A,
                        (q + 9 * (jj - 1)) + 2);
        }
      }
      for (ii = qp1; ii < 6; ii++) {
        Vf[(ii + 5 * q) - 1] = e[ii - 1];
      }
    }
  }
  m = 3;
  e[3] = b_A[39];
  e[4] = 0.0;
  for (int q{4}; q >= 0; q--) {
    qp1 = q + 2;
    qq = q + 9 * q;
    if (b_s[q] != 0.0) {
      for (jj = qp1; jj < 6; jj++) {
        i = (q + 9 * (jj - 1)) + 1;
        blas::xaxpy(9 - q, -(blas::xdotc(9 - q, U, qq + 1, U, i) / U[qq]),
                    qq + 1, U, i);
      }
      for (ii = q + 1; ii < 10; ii++) {
        i = (ii + 9 * q) - 1;
        U[i] = -U[i];
      }
      U[qq]++;
      for (ii = 0; ii < q; ii++) {
        U[ii + 9 * q] = 0.0;
      }
    } else {
      std::memset(&U[q * 9], 0, 9U * sizeof(double));
      U[qq] = 1.0;
    }
  }
  for (int q{4}; q >= 0; q--) {
    if ((q + 1 <= 3) && (e[q] != 0.0)) {
      qp1 = q + 2;
      i = (q + 5 * q) + 2;
      for (jj = qp1; jj < 6; jj++) {
        qp1jj = (q + 5 * (jj - 1)) + 2;
        blas::b_xaxpy(4 - q,
                      -(blas::b_xdotc(4 - q, Vf, i, Vf, qp1jj) / Vf[i - 1]), i,
                      Vf, qp1jj);
      }
    }
    for (ii = 0; ii < 5; ii++) {
      Vf[ii + 5 * q] = 0.0;
    }
    Vf[q + 5 * q] = 1.0;
  }
  qq = 0;
  snorm = 0.0;
  for (int q{0}; q < 5; q++) {
    nrm = b_s[q];
    if (nrm != 0.0) {
      rt = std::abs(nrm);
      nrm /= rt;
      b_s[q] = rt;
      if (q + 1 < 5) {
        e[q] /= nrm;
      }
      i = 9 * q;
      jj = i + 9;
      for (int k{i + 1}; k <= jj; k++) {
        U[k - 1] *= nrm;
      }
    }
    if (q + 1 < 5) {
      nrm = e[q];
      if (nrm != 0.0) {
        rt = std::abs(nrm);
        nrm = rt / nrm;
        e[q] = rt;
        b_s[q + 1] *= nrm;
        i = 5 * (q + 1);
        jj = i + 5;
        for (int k{i + 1}; k <= jj; k++) {
          Vf[k - 1] *= nrm;
        }
      }
    }
    snorm = std::fmax(snorm, std::fmax(std::abs(b_s[q]), std::abs(e[q])));
  }
  while ((m + 2 > 0) && (qq < 75)) {
    bool exitg1;
    jj = m + 1;
    ii = m + 1;
    exitg1 = false;
    while (!(exitg1 || (ii == 0))) {
      nrm = std::abs(e[ii - 1]);
      if ((nrm <= 2.2204460492503131E-16 *
                      (std::abs(b_s[ii - 1]) + std::abs(b_s[ii]))) ||
          (nrm <= 1.0020841800044864E-292) ||
          ((qq > 20) && (nrm <= 2.2204460492503131E-16 * snorm))) {
        e[ii - 1] = 0.0;
        exitg1 = true;
      } else {
        ii--;
      }
    }
    if (ii == m + 1) {
      i = 4;
    } else {
      qp1jj = m + 2;
      i = m + 2;
      exitg1 = false;
      while ((!exitg1) && (i >= ii)) {
        qp1jj = i;
        if (i == ii) {
          exitg1 = true;
        } else {
          nrm = 0.0;
          if (i < m + 2) {
            nrm = std::abs(e[i - 1]);
          }
          if (i > ii + 1) {
            nrm += std::abs(e[i - 2]);
          }
          rt = std::abs(b_s[i - 1]);
          if ((rt <= 2.2204460492503131E-16 * nrm) ||
              (rt <= 1.0020841800044864E-292)) {
            b_s[i - 1] = 0.0;
            exitg1 = true;
          } else {
            i--;
          }
        }
      }
      if (qp1jj == ii) {
        i = 3;
      } else if (qp1jj == m + 2) {
        i = 1;
      } else {
        i = 2;
        ii = qp1jj;
      }
    }
    switch (i) {
    case 1: {
      rt = e[m];
      e[m] = 0.0;
      for (int k{jj}; k >= ii + 1; k--) {
        double sm;
        sm = blas::xrotg(b_s[k - 1], rt, sqds);
        if (k > ii + 1) {
          double b;
          b = e[k - 2];
          rt = -sqds * b;
          e[k - 2] = b * sm;
        }
        blas::xrot(Vf, 5 * (k - 1) + 1, 5 * (m + 1) + 1, sm, sqds);
      }
    } break;
    case 2: {
      rt = e[ii - 1];
      e[ii - 1] = 0.0;
      for (int k{ii + 1}; k <= m + 2; k++) {
        double b;
        double sm;
        sm = blas::xrotg(b_s[k - 1], rt, sqds);
        b = e[k - 1];
        rt = -sqds * b;
        e[k - 1] = b * sm;
        blas::b_xrot(U, 9 * (k - 1) + 1, 9 * (ii - 1) + 1, sm, sqds);
      }
    } break;
    case 3: {
      double b;
      double scale;
      double sm;
      nrm = b_s[m + 1];
      scale = std::fmax(
          std::fmax(std::fmax(std::fmax(std::abs(nrm), std::abs(b_s[m])),
                              std::abs(e[m])),
                    std::abs(b_s[ii])),
          std::abs(e[ii]));
      sm = nrm / scale;
      nrm = b_s[m] / scale;
      rt = e[m] / scale;
      sqds = b_s[ii] / scale;
      b = ((nrm + sm) * (nrm - sm) + rt * rt) / 2.0;
      nrm = sm * rt;
      nrm *= nrm;
      if ((b != 0.0) || (nrm != 0.0)) {
        rt = std::sqrt(b * b + nrm);
        if (b < 0.0) {
          rt = -rt;
        }
        rt = nrm / (b + rt);
      } else {
        rt = 0.0;
      }
      rt += (sqds + sm) * (sqds - sm);
      nrm = sqds * (e[ii] / scale);
      for (int k{ii + 1}; k <= jj; k++) {
        sm = blas::xrotg(rt, nrm, sqds);
        if (k > ii + 1) {
          e[k - 2] = rt;
        }
        nrm = e[k - 1];
        b = b_s[k - 1];
        e[k - 1] = sm * nrm - sqds * b;
        rt = sqds * b_s[k];
        b_s[k] *= sm;
        blas::xrot(Vf, 5 * (k - 1) + 1, 5 * k + 1, sm, sqds);
        b_s[k - 1] = sm * b + sqds * nrm;
        sm = blas::xrotg(b_s[k - 1], rt, sqds);
        b = e[k - 1];
        rt = sm * b + sqds * b_s[k];
        b_s[k] = -sqds * b + sm * b_s[k];
        nrm = sqds * e[k];
        e[k] *= sm;
        blas::b_xrot(U, 9 * (k - 1) + 1, 9 * k + 1, sm, sqds);
      }
      e[m] = rt;
      qq++;
    } break;
    default:
      if (b_s[ii] < 0.0) {
        b_s[ii] = -b_s[ii];
        i = 5 * ii;
        jj = i + 5;
        for (int k{i + 1}; k <= jj; k++) {
          Vf[k - 1] = -Vf[k - 1];
        }
      }
      qp1 = ii + 1;
      while ((ii + 1 < 5) && (b_s[ii] < b_s[qp1])) {
        rt = b_s[ii];
        b_s[ii] = b_s[qp1];
        b_s[qp1] = rt;
        blas::xswap(Vf, 5 * ii + 1, 5 * (ii + 1) + 1);
        blas::b_xswap(U, 9 * ii + 1, 9 * (ii + 1) + 1);
        ii = qp1;
        qp1++;
      }
      qq = 0;
      m--;
      break;
    }
  }
  for (int k{0}; k < 5; k++) {
    s[k] = b_s[k];
    for (i = 0; i < 5; i++) {
      qp1jj = i + 5 * k;
      V[qp1jj] = Vf[qp1jj];
    }
  }
}

} // namespace internal
} // namespace coder

//
// File trailer for svd.cpp
//
// [EOF]
//
