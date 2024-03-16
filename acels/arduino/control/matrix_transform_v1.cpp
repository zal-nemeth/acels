//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: matrix_transform_v1.cpp
//
// MATLAB Coder version            : 5.2
// C/C++ source code generated on  : 10-Apr-2021 12:17:27
//

// Include Files
#include "matrix_transform_v1.h"
#include "CompactRegressionTree.h"
#include "matrix_transform_v1_data.h"
#include "rt_nonfinite.h"
#include "svd.h"
#include "table.h"
#include "varNamesDim.h"
#include "coder_bounded_array.h"
#include "rt_defines.h"
#include <cmath>
#include <cstring>
#include <math.h>

// Function Declarations
static double rt_atan2d_snf(double u0, double u1);

// Function Definitions
//
// Arguments    : double u0
//                double u1
// Return Type  : double
//
static double rt_atan2d_snf(double u0, double u1)
{
  double y;
  if (std::isnan(u0) || std::isnan(u1)) {
    y = rtNaN;
  } else if (std::isinf(u0) && std::isinf(u1)) {
    int b_u0;
    int b_u1;
    if (u0 > 0.0) {
      b_u0 = 1;
    } else {
      b_u0 = -1;
    }
    if (u1 > 0.0) {
      b_u1 = 1;
    } else {
      b_u1 = -1;
    }
    y = std::atan2(static_cast<double>(b_u0), static_cast<double>(b_u1));
  } else if (u1 == 0.0) {
    if (u0 > 0.0) {
      y = RT_PI / 2.0;
    } else if (u0 < 0.0) {
      y = -(RT_PI / 2.0);
    } else {
      y = 0.0;
    }
  } else {
    y = std::atan2(u0, u1);
  }
  return y;
}

//
// Arguments    : double x
//                double y
//                double z
//                double F_x
//                double F_y
//                double F_z
//                double T_x
//                double T_y
//                double b_I[9]
// Return Type  : void
//
void matrix_transform_v1(double x, double y, double z, double F_x, double F_y,
                         double F_z, double T_x, double T_y, double b_I[9])
{
  static const double dv4[119]{
      4.5,  5.5,  8.5,  2.5,  1.5,  4.5,  4.5,  0.5,  1.5,  7.5,  2.5,  1.5,
      6.5,  1.5,  11.5, 0.0,  1.5,  0.0,  0.0,  0.0,  17.5, 9.5,  7.5,  0.0,
      3.5,  7.5,  7.5,  0.5,  11.5, 7.5,  13.5, 0.0,  0.0,  0.5,  0.0,  0.0,
      16.5, 0.0,  17.5, 0.0,  0.0,  0.0,  17.5, 0.0,  17.5, 0.0,  0.0,  0.0,
      2.5,  0.0,  9.5,  6.5,  7.5,  11.5, 11.5, 0.0,  0.0,  3.5,  0.0,  5.5,
      0.0,  7.5,  0.0,  0.0,  0.0,  16.5, 17.5, 0.0,  17.5, 0.0,  17.5, 0.0,
      0.0,  0.0,  0.0,  11.5, 11.5, 11.5, 11.5, 11.5, 11.5, 0.0,  0.0,  10.5,
      0.0,  12.5, 0.0,  14.5, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
      0.0,  0.0,  0.0,  0.0,  0.0,  11.5, 11.5, 10.5, 10.5, 11.5, 11.5, 0.0,
      0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0};
  static const double dv5[119]{2.985119047590389,
                               5.3985714285377462,
                               1.8880952380688583,
                               2.1883333332153985,
                               6.682666666666675,
                               2.8464285713831563,
                               1.3404761904606817,
                               0.89666666643079984,
                               3.4799999999999982,
                               8.12333333333333,
                               5.7222222222222188,
                               0.94499999980924942,
                               3.4406250000000012,
                               0.44685714279199962,
                               1.6197321428571432,
                               -7.0760000000000012E-10,
                               1.3450000000000002,
                               4.1999999999999993,
                               3.0,
                               6.5000000000000009,
                               8.3730769230769226,
                               6.546666666666666,
                               5.3099999999999969,
                               0.24249999952312498,
                               1.4133333333333333,
                               3.9218749999999987,
                               2.9593749999999988,
                               0.11357142840857143,
                               0.66904761904761856,
                               2.0916666666666659,
                               1.2657812499999999,
                               0.90999999999999992,
                               1.7800000000000002,
                               8.8049999999999979,
                               6.9333333333333336,
                               5.8500000000000005,
                               6.8000000000000007,
                               4.275,
                               5.4692307692307649,
                               1.1825,
                               1.8749999999999998,
                               3.0333333333333332,
                               4.1269230769230765,
                               2.2833333333333332,
                               3.1153846153846128,
                               -3.2571428571428573E-10,
                               0.22714285714285717,
                               0.85888888888888881,
                               0.52666666666666673,
                               1.6222222222222222,
                               2.1999999999999993,
                               1.4531249999999991,
                               1.0784374999999995,
                               9.4300000000000033,
                               8.18,
                               7.3285714285714292,
                               5.875,
                               5.729999999999996,
                               4.6000000000000005,
                               4.3149999999999986,
                               3.5,
                               3.2549999999999977,
                               2.65,
                               0.35750000000000004,
                               0.61125,
                               2.5153846153846149,
                               2.0423076923076922,
                               1.0250000000000001,
                               1.5142857142857129,
                               0.83666666666666678,
                               1.1342307692307689,
                               9.05,
                               9.6833333333333336,
                               7.8500000000000005,
                               8.4,
                               6.1399999999999988,
                               5.32,
                               4.63,
                               4.0,
                               3.5000000000000004,
                               3.01,
                               2.6444444444444444,
                               2.2249999999999996,
                               2.1299999999999994,
                               1.75,
                               1.5727272727272714,
                               1.2999999999999998,
                               1.1884999999999997,
                               0.95333333333333337,
                               5.925,
                               6.2833333333333332,
                               5.125,
                               5.4499999999999993,
                               4.45,
                               4.75,
                               3.85,
                               4.1000000000000005,
                               3.375,
                               3.583333333333333,
                               2.875,
                               3.1,
                               2.28,
                               1.9799999999999998,
                               1.6818181818181821,
                               1.4636363636363638,
                               1.2800000000000002,
                               1.097,
                               2.175,
                               2.3499999999999996,
                               1.8999999999999997,
                               2.0333333333333337,
                               1.5499999999999998,
                               1.7571428571428569,
                               1.3499999999999999,
                               1.5285714285714285,
                               1.225,
                               1.3166666666666667,
                               1.0425,
                               1.1333333333333333};
  static const double dv2[101]{
      5.5,  12.5, 12.5, 2.5,  16.5, 9.5,  9.5,  8.5,  7.5,  2.5, 18.5, 7.5,
      12.5, 16.5, 16.5, 0.5,  0.5,  3.5,  3.5,  14.5, 14.5, 2.5, 2.5,  7.5,
      7.5,  7.5,  7.5,  7.5,  18.5, 12.5, 12.5, 0.0,  1.5,  0.0, 0.0,  0.0,
      4.5,  0.0,  10.5, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 0.0,  6.5,
      10.5, 8.5,  10.5, 10.5, 10.5, 13.5, 13.5, 0.0,  0.0,  0.0, 0.0,  14.5,
      14.5, 18.5, 18.5, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 0.0,  0.0,
      0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  11.5, 0.0,  10.5, 0.0, 14.5, 0.0,
      10.5, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 0.0,  0.0,
      0.0,  0.0,  0.0,  0.0,  0.0};
  static const double dv3[101]{0.60975892857143066,
                               0.84979365079365266,
                               0.46573809523809634,
                               1.0030769230769243,
                               0.6007083333333334,
                               0.539400000000001,
                               0.34603750000000033,
                               1.125999999999999,
                               0.8801538461538454,
                               0.71174999999999966,
                               0.48966666666666625,
                               0.66567307692307687,
                               0.45521794871794924,
                               0.41599999999999976,
                               0.29939583333333342,
                               1.1789629629629621,
                               1.0068333333333332,
                               0.931916666666666,
                               0.79733333333333323,
                               0.79808333333333326,
                               0.62541666666666662,
                               0.54708333333333314,
                               0.43224999999999997,
                               0.71823076923076867,
                               0.61311538461538417,
                               0.50620512820512786,
                               0.40423076923076906,
                               0.48031249999999986,
                               0.35168749999999982,
                               0.33879166666666649,
                               0.2599999999999999,
                               1.2758888888888891,
                               1.1304999999999994,
                               1.0947500000000001,
                               0.962875,
                               1.0084999999999997,
                               0.89362499999999978,
                               0.8638,
                               0.7641,
                               0.85850000000000015,
                               0.73766666666666658,
                               0.6695,
                               0.58133333333333337,
                               0.60649999999999993,
                               0.48766666666666669,
                               0.47216666666666662,
                               0.39233333333333337,
                               0.76074999999999982,
                               0.6502,
                               0.6488124999999999,
                               0.55600000000000016,
                               0.53470833333333312,
                               0.46059999999999995,
                               0.4257916666666664,
                               0.3697333333333333,
                               0.51562500000000011,
                               0.44499999999999995,
                               0.38437499999999997,
                               0.319,
                               0.37333333333333335,
                               0.30425,
                               0.28308333333333335,
                               0.23691666666666664,
                               1.1764444444444444,
                               1.0845555555555557,
                               0.92975,
                               0.85749999999999993,
                               0.79533333333333334,
                               0.71725,
                               0.7912499999999999,
                               0.73025,
                               0.67633333333333334,
                               0.611,
                               0.6745,
                               0.62312499999999993,
                               0.57766666666666677,
                               0.52350000000000008,
                               0.5764999999999999,
                               0.5138125,
                               0.4956,
                               0.4431,
                               0.457875,
                               0.40974999999999989,
                               0.39659999999999995,
                               0.3563,
                               0.39516666666666667,
                               0.35149999999999992,
                               0.3205,
                               0.288,
                               0.30600000000000005,
                               0.26016666666666671,
                               0.25400000000000006,
                               0.21983333333333335,
                               0.53350000000000009,
                               0.49412499999999993,
                               0.4595,
                               0.4185,
                               0.42487500000000006,
                               0.394625,
                               0.36850000000000005,
                               0.33799999999999997};
  static const double dv[91]{
      7.5,  3.5,  5.5,  1.5,  5.5,  12.5, 9.5,  0.5,  6.5,  5.5, 9.5, 2.5, 2.5,
      12.5, 12.5, 0.0,  6.5,  2.5,  2.5,  2.5,  2.5,  5.5,  5.5, 9.5, 9.5, 0.5,
      3.5,  9.5,  7.5,  12.5, 12.5, 0.0,  0.0,  0.0,  0.0,  0.0, 0.0, 0.0, 0.0,
      0.0,  0.0,  0.0,  0.0,  12.5, 12.5, 0.0,  0.0,  0.0,  0.0, 0.0, 1.5, 0.0,
      4.5,  0.0,  7.5,  6.5,  8.5,  9.5,  9.5,  10.5, 13.5, 0.0, 0.0, 0.0, 0.0,
      0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 0.0, 0.0, 0.0,
      0.0,  0.0,  15.5, 0.0,  15.5, 0.0,  11.5, 0.0,  14.5, 0.0, 0.0, 0.0, 0.0};
  static const double dv1[91]{0.25459821428747159,
                              0.10496093750461115,
                              0.34668269230769322,
                              0.045500000009221914,
                              0.16442187500000011,
                              0.50679487179487215,
                              0.25061538461538518,
                              0.015250000018443739,
                              0.075749999999999942,
                              0.23666666666666655,
                              0.12107499999999993,
                              0.4126666666666664,
                              0.56562499999999971,
                              0.31730769230769229,
                              0.20615384615384641,
                              3.6887499999999993E-11,
                              0.030499999999999992,
                              0.10357142857142855,
                              0.054111111111111096,
                              0.19416666666666665,
                              0.27916666666666667,
                              0.15374999999999997,
                              0.099291666666666611,
                              0.46799999999999992,
                              0.35733333333333328,
                              0.64666666666666628,
                              0.48458333333333314,
                              0.26199999999999984,
                              0.35187499999999977,
                              0.1699999999999999,
                              0.22874999999999995,
                              0.041857142857142857,
                              0.021666666666666664,
                              0.083285714285714282,
                              0.12385714285714286,
                              0.043444444444444445,
                              0.064777777777777767,
                              0.21666666666666667,
                              0.17166666666666666,
                              0.3133333333333333,
                              0.245,
                              0.1275,
                              0.18000000000000002,
                              0.082583333333333342,
                              0.11599999999999999,
                              0.40333333333333338,
                              0.51111111111111118,
                              0.31333333333333341,
                              0.38666666666666666,
                              0.7125,
                              0.61374999999999991,
                              0.53125,
                              0.46124999999999994,
                              0.23124999999999998,
                              0.2825,
                              0.38312499999999994,
                              0.32062499999999988,
                              0.19199999999999998,
                              0.148,
                              0.25749999999999995,
                              0.1999999999999999,
                              0.092833333333333337,
                              0.072333333333333333,
                              0.13,
                              0.102,
                              0.64375,
                              0.58375,
                              0.4825,
                              0.43999999999999995,
                              0.30666666666666659,
                              0.25833333333333336,
                              0.40000000000000008,
                              0.36624999999999996,
                              0.33499999999999996,
                              0.30625,
                              0.16833333333333333,
                              0.20777777777777778,
                              0.12833333333333335,
                              0.16111111111111112,
                              0.28,
                              0.24624999999999997,
                              0.2175,
                              0.19124999999999995,
                              0.23,
                              0.256,
                              0.17833333333333334,
                              0.19900000000000004,
                              0.266,
                              0.246,
                              0.20799999999999996,
                              0.19};
  static const signed char iv9[238]{
      2,   4,   6,  8,  10,  12,  14, 16,  18,  20,  22,  24,  26,  28,  30,
      0,   32,  0,  0,  0,   34,  36, 38,  0,   40,  42,  44,  46,  48,  50,
      52,  0,   0,  54, 0,   0,   56, 0,   58,  0,   0,   0,   60,  0,   62,
      0,   0,   0,  64, 0,   66,  68, 70,  72,  74,  0,   0,   76,  0,   78,
      0,   80,  0,  0,  0,   82,  84, 0,   86,  0,   88,  0,   0,   0,   0,
      90,  92,  94, 96, 98,  100, 0,  0,   102, 0,   104, 0,   106, 0,   0,
      0,   0,   0,  0,  0,   0,   0,  0,   0,   0,   0,   108, 110, 112, 114,
      116, 118, 0,  0,  0,   0,   0,  0,   0,   0,   0,   0,   0,   0,   3,
      5,   7,   9,  11, 13,  15,  17, 19,  21,  23,  25,  27,  29,  31,  0,
      33,  0,   0,  0,  35,  37,  39, 0,   41,  43,  45,  47,  49,  51,  53,
      0,   0,   55, 0,  0,   57,  0,  59,  0,   0,   0,   61,  0,   63,  0,
      0,   0,   65, 0,  67,  69,  71, 73,  75,  0,   0,   77,  0,   79,  0,
      81,  0,   0,  0,  83,  85,  0,  87,  0,   89,  0,   0,   0,   0,   91,
      93,  95,  97, 99, 101, 0,   0,  103, 0,   105, 0,   107, 0,   0,   0,
      0,   0,   0,  0,  0,   0,   0,  0,   0,   0,   109, 111, 113, 115, 117,
      119, 0,   0,  0,  0,   0,   0,  0,   0,   0,   0,   0,   0};
  static const signed char iv6[202]{
      2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32,  34,
      36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 0,  64,  0,
      0,  0,  66, 0,  68, 0,  0,  0,  0,  0,  0,  0,  0,  70, 72, 74,  76,
      78, 80, 82, 84, 0,  0,  0,  0,  86, 88, 90, 92, 0,  0,  0,  0,   0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  94, 0,  96, 0,  98, 0,   100,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,   3,
      5,  7,  9,  11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35,  37,
      39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 0,  65, 0,   0,
      0,  67, 0,  69, 0,  0,  0,  0,  0,  0,  0,  0,  71, 73, 75, 77,  79,
      81, 83, 85, 0,  0,  0,  0,  87, 89, 91, 93, 0,  0,  0,  0,  0,   0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  95, 0,  97, 0,  99, 0,  101, 0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0};
  static const signed char iv3[182]{
      2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 0,  32,
      34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  62, 64, 0,  0,  0,  0,  0,  66,
      0,  68, 0,  70, 72, 74, 76, 78, 80, 82, 0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  84, 0,  86, 0,  88,
      0,  90, 0,  0,  0,  0,  3,  5,  7,  9,  11, 13, 15, 17, 19, 21, 23,
      25, 27, 29, 31, 0,  33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55,
      57, 59, 61, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  63, 65,
      0,  0,  0,  0,  0,  67, 0,  69, 0,  71, 73, 75, 77, 79, 81, 83, 0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  85, 0,  87, 0,  89, 0,  91, 0,  0,  0,  0};
  static const signed char iv10[119]{
      56, 55, 54, 51, 53, 52, 50, 43, 40, 46, 48, 44, 47, 35, 49, 0,  30,
      0,  0,  0,  46, 39, 41, 0,  28, 42, 38, 12, 22, 34, 33, 0,  0,  45,
      0,  0,  39, 0,  41, 0,  0,  0,  36, 0,  29, 0,  0,  0,  11, 0,  31,
      25, 19, 26, 24, 0,  0,  37, 0,  32, 0,  27, 0,  0,  0,  21, 23, 0,
      18, 0,  15, 0,  0,  0,  0,  17, 14, 13, 9,  6,  8,  0,  0,  20, 0,
      16, 0,  10, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  4,
      3,  7,  5,  2,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0};
  static const signed char iv8[119]{
      2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 0, 0, 0, 1, 1, 1, 0,
      1, 1, 1, 1, 2, 1, 2, 0, 0, 2, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0,
      1, 0, 2, 1, 1, 1, 1, 0, 0, 2, 0, 2, 0, 2, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0,
      0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 2, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  static const signed char iv5[101]{
      2, 1, 1, 2, 1, 2, 2, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 2, 2, 1, 1,
      2, 2, 1, 1, 1, 1, 2, 1, 2, 2, 0, 2, 0, 0, 0, 2, 0, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 2, 1, 2, 1, 2, 2, 2, 2, 0, 0, 0, 0, 1, 1, 1, 1,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 2, 0,
      1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  static const signed char iv7[101]{
      50, 49, 48, 46, 45, 47, 44, 43, 40, 41, 36, 39, 42, 38, 34, 37, 30,
      33, 26, 29, 23, 28, 19, 35, 32, 31, 25, 20, 18, 24, 15, 0,  27, 0,
      0,  0,  21, 0,  16, 0,  0,  0,  0,  0,  0,  0,  0,  17, 12, 13, 10,
      22, 11, 14, 6,  0,  0,  0,  0,  7,  2,  9,  3,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  8,  0,  5,  0,  4,  0,  1,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0};
  static const signed char iv2[91]{
      1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 2, 1, 1, 0, 2, 1, 1, 2, 2, 1, 1,
      1, 1, 2, 2, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0,
      0, 0, 0, 0, 2, 0, 2, 0, 2, 2, 2, 1, 1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 2, 0, 2, 0, 0, 0, 0};
  static const signed char iv4[91]{
      44, 42, 43, 34, 38, 41, 40, 19, 25, 32, 28, 36, 39, 37, 35, 0, 5,  14, 6,
      15, 22, 20, 16, 31, 26, 33, 27, 21, 29, 24, 30, 0,  0,  0,  0, 0,  0,  0,
      0,  0,  0,  0,  0,  3,  7,  0,  0,  0,  0,  0,  23, 0,  18, 0, 17, 12, 9,
      13, 11, 15, 10, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 0,  0,  0,
      0,  0,  0,  0,  8,  0,  4,  0,  2,  0,  1,  0,  0,  0,  0};
  static const signed char iv[9]{-30, 0, 30, -30, 0, 30, -30, 0, 30};
  static const signed char iv1[9]{30, 30, 30, 0, 0, 0, -30, -30, -30};
  static const bool bv2[119]{
      false, false, false, false, false, false, false, false, false, false,
      false, false, false, false, false, true,  false, true,  true,  true,
      false, false, false, true,  false, false, false, false, false, false,
      false, true,  true,  false, true,  true,  false, true,  false, true,
      true,  true,  false, true,  false, true,  true,  true,  false, true,
      false, false, false, false, false, true,  true,  false, true,  false,
      true,  false, true,  true,  true,  false, false, true,  false, true,
      false, true,  true,  true,  true,  false, false, false, false, false,
      false, true,  true,  false, true,  false, true,  false, true,  true,
      true,  true,  true,  true,  true,  true,  true,  true,  true,  true,
      true,  false, false, false, false, false, false, true,  true,  true,
      true,  true,  true,  true,  true,  true,  true,  true,  true};
  static const bool bv1[101]{
      false, false, false, false, false, false, false, false, false, false,
      false, false, false, false, false, false, false, false, false, false,
      false, false, false, false, false, false, false, false, false, false,
      false, true,  false, true,  true,  true,  false, true,  false, true,
      true,  true,  true,  true,  true,  true,  true,  false, false, false,
      false, false, false, false, false, true,  true,  true,  true,  false,
      false, false, false, true,  true,  true,  true,  true,  true,  true,
      true,  true,  true,  true,  true,  true,  true,  true,  false, true,
      false, true,  false, true,  false, true,  true,  true,  true,  true,
      true,  true,  true,  true,  true,  true,  true,  true,  true,  true,
      true};
  static const bool bv[91]{
      false, false, false, false, false, false, false, false, false, false,
      false, false, false, false, false, true,  false, false, false, false,
      false, false, false, false, false, false, false, false, false, false,
      false, true,  true,  true,  true,  true,  true,  true,  true,  true,
      true,  true,  true,  false, false, true,  true,  true,  true,  true,
      false, true,  false, true,  false, false, false, false, false, false,
      false, true,  true,  true,  true,  true,  true,  true,  true,  true,
      true,  true,  true,  true,  true,  true,  true,  true,  true,  true,
      false, true,  false, true,  false, true,  false, true,  true,  true,
      true};
  coder::classreg::learning::regr::CompactRegressionTree obj;
  coder::classreg::learning::regr::b_CompactRegressionTree b_obj;
  coder::classreg::learning::regr::c_CompactRegressionTree c_obj;
  coder::table b_input_data;
  coder::table input_data;
  double A[45];
  double C[45];
  double U[45];
  double b_A[45];
  double V[25];
  double s[5];
  int exponent;
  std::memset(&A[0], 0, 45U * sizeof(double));
  input_data.data[1] = z;
  input_data.arrayProps.Description.size[0] = 1;
  input_data.arrayProps.Description.size[1] = 0;
  input_data.varDim.hasUnits = false;
  input_data.varDim.hasDescrs = false;
  input_data.varDim.hasContinuity = false;
  obj.ResponseTransform = coder::classreg::learning::coderutils::Identity;
  b_obj.ResponseTransform = coder::classreg::learning::coderutils::Identity;
  c_obj.ResponseTransform = coder::classreg::learning::coderutils::Identity;
  input_data.varDim.continuity[0] =
      coder::matlab::internal::coder::tabular::unset;
  input_data.varDim.continuity[1] =
      coder::matlab::internal::coder::tabular::unset;
  for (int i{0}; i < 9; i++) {
    double a_tmp;
    double f_x;
    double f_z;
    double theta_i;
    int b_i;
    int c_i;
    bool p;
    theta_i = x - static_cast<double>(iv[i]);
    a_tmp = y - static_cast<double>(iv1[i]);
    input_data.data[0] = std::sqrt(theta_i * theta_i + a_tmp * a_tmp);
    theta_i = rt_atan2d_snf(a_tmp, theta_i);
    input_data.varDim.descrs[0].f1.size[0] = 1;
    input_data.varDim.descrs[0].f1.size[1] = 0;
    input_data.varDim.descrs[1].f1.size[0] = 1;
    input_data.varDim.descrs[1].f1.size[1] = 0;
    input_data.varDim.units[0] = input_data.varDim.descrs[0];
    input_data.varDim.units[1] = input_data.varDim.descrs[1];
    input_data.varDim.descrs[0].f1.size[0] = 1;
    input_data.varDim.descrs[0].f1.size[1] = 0;
    input_data.varDim.descrs[1].f1.size[0] = 1;
    input_data.varDim.descrs[1].f1.size[1] = 0;
    obj.PruneList.size[0] = 91;
    for (b_i = 0; b_i < 91; b_i++) {
      obj.CutPoint[b_i] = dv[b_i];
      obj.CutPredictorIndex[b_i] = iv2[b_i];
      c_i = b_i << 1;
      obj.Children[c_i] = iv3[b_i];
      obj.Children[c_i + 1] = iv3[b_i + 91];
      obj.NanCutPoints[b_i] = bv[b_i];
      obj.InfCutPoints[b_i] = false;
      obj.PruneList.data[b_i] = iv4[b_i];
      obj.NodeMean[b_i] = dv1[b_i];
    }
    f_x = obj.predict(&input_data);
    input_data.parenReference(&b_input_data);
    b_obj.PruneList.size[0] = 101;
    for (b_i = 0; b_i < 101; b_i++) {
      b_obj.CutPoint[b_i] = dv2[b_i];
      b_obj.CutPredictorIndex[b_i] = iv5[b_i];
      c_i = b_i << 1;
      b_obj.Children[c_i] = iv6[b_i];
      b_obj.Children[c_i + 1] = iv6[b_i + 101];
      b_obj.NanCutPoints[b_i] = bv1[b_i];
      b_obj.InfCutPoints[b_i] = false;
      b_obj.PruneList.data[b_i] = iv7[b_i];
      b_obj.NodeMean[b_i] = dv3[b_i];
    }
    f_z = b_obj.predict(&input_data);
    input_data.parenReference(&b_input_data);
    c_obj.PruneList.size[0] = 119;
    for (b_i = 0; b_i < 119; b_i++) {
      c_obj.CutPoint[b_i] = dv4[b_i];
      c_obj.CutPredictorIndex[b_i] = iv8[b_i];
      c_i = b_i << 1;
      c_obj.Children[c_i] = iv9[b_i];
      c_obj.Children[c_i + 1] = iv9[b_i + 119];
      c_obj.NanCutPoints[b_i] = bv2[b_i];
      c_obj.InfCutPoints[b_i] = false;
      c_obj.PruneList.data[b_i] = iv10[b_i];
      c_obj.NodeMean[b_i] = dv5[b_i];
    }
    double A_tmp;
    double t_y;
    t_y = c_obj.predict(&input_data);
    A_tmp = std::sin(theta_i);
    a_tmp = std::cos(theta_i);
    A[5 * i] = a_tmp * f_x;
    A[5 * i + 1] = A_tmp * ((input_data.data[0] * -2.9925868506493495E-13 +
                             -3.3209443118156367E-12) +
                            z * 5.2335574229691879E-13);
    A[5 * i + 2] = f_z;
    A[5 * i + 3] = -A_tmp * ((input_data.data[0] * -5.4728238636363631E-12 +
                              -1.8763588940667148E-11) +
                             z * 1.9186184161064423E-11);
    A[5 * i + 4] = a_tmp * t_y;
    for (c_i = 0; c_i < 5; c_i++) {
      for (b_i = 0; b_i < 9; b_i++) {
        b_A[b_i + 9 * c_i] = A[c_i + 5 * b_i];
      }
    }
    p = true;
    for (b_i = 0; b_i < 45; b_i++) {
      C[b_i] = 0.0;
      if ((!p) || (std::isinf(b_A[b_i]) || std::isnan(b_A[b_i]))) {
        p = false;
      }
    }
    if (!p) {
      for (c_i = 0; c_i < 45; c_i++) {
        C[c_i] = rtNaN;
      }
    } else {
      int r;
      coder::internal::svd(b_A, U, s, V);
      theta_i = std::abs(s[0]);
      if ((!std::isinf(theta_i)) && (!std::isnan(theta_i))) {
        if (theta_i <= 2.2250738585072014E-308) {
          theta_i = 4.94065645841247E-324;
        } else {
          frexp(theta_i, &exponent);
          theta_i = std::ldexp(1.0, exponent - 53);
        }
      } else {
        theta_i = rtNaN;
      }
      theta_i *= 9.0;
      r = -1;
      b_i = 0;
      while ((b_i < 5) && (s[b_i] > theta_i)) {
        r++;
        b_i++;
      }
      if (r + 1 > 0) {
        int br;
        int vcol;
        vcol = 1;
        for (br = 0; br <= r; br++) {
          theta_i = 1.0 / s[br];
          c_i = vcol + 4;
          for (b_i = vcol; b_i <= c_i; b_i++) {
            V[b_i - 1] *= theta_i;
          }
          vcol += 5;
        }
        for (vcol = 0; vcol <= 40; vcol += 5) {
          c_i = vcol + 1;
          b_i = vcol + 5;
          if (c_i <= b_i) {
            std::memset(&C[c_i + -1], 0, ((b_i - c_i) + 1) * sizeof(double));
          }
        }
        br = 0;
        for (vcol = 0; vcol <= 40; vcol += 5) {
          int ar;
          ar = -1;
          br++;
          c_i = br + 9 * r;
          for (int ib{br}; ib <= c_i; ib += 9) {
            int i1;
            b_i = vcol + 1;
            i1 = vcol + 5;
            for (int ic{b_i}; ic <= i1; ic++) {
              C[ic - 1] += U[ib - 1] * V[(ar + ic) - vcol];
            }
            ar += 5;
          }
        }
      }
    }
    s[0] = F_x;
    s[1] = F_y;
    s[2] = F_z;
    s[3] = T_x;
    s[4] = T_y;
    for (c_i = 0; c_i < 9; c_i++) {
      theta_i = 0.0;
      for (b_i = 0; b_i < 5; b_i++) {
        theta_i += C[b_i + 5 * c_i] * s[b_i];
      }
      b_I[c_i] = theta_i;
    }
  }
}

//
// File trailer for matrix_transform_v1.cpp
//
// [EOF]
//
