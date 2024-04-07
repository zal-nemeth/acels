/*  A.C.E.L.S. Control Algorithm
 *  Zalan Nemeth
 */

// Includeing Packages/Dependencies
#include "Arduino.h"
#include <PID_v1.h>
#include <SimpleKalmanFilter.h>
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Include built dependencies
#include "main_functions.h"
#include "constants.h"
#include "model.h"
#include "matrix_transform.h"
#include "matrix_transform_initialize.h"
#include "matrix_transform_terminate.h"

// Defining Kalman Filter variable for each sensor
SimpleKalmanFilter filter1(5, 5, 0.01);
SimpleKalmanFilter filter2(5, 5, 0.01);
SimpleKalmanFilter filter3(5, 5, 0.01);
SimpleKalmanFilter filter4(5, 5, 0.01);
SimpleKalmanFilter filter5(5, 5, 0.01);
SimpleKalmanFilter filter6(5, 5, 0.01);
SimpleKalmanFilter filter7(5, 5, 0.01);
SimpleKalmanFilter filter8(5, 5, 0.01);

// ----------------------------------------------------------------------
// Include data statistics from NN code
// Change mean and std values based on dataset used for model training
// ----------------------------------------------------------------------
// Extended Dataset Statistics
float mean_s1 = 431.5683288574219;
float mean_s2 = 352.7138366699219;
float mean_s3 = 402.7107238769531;
float mean_s4 = 350.7831726074219;
float mean_s5 = 331.1814270019531;
float mean_s6 = 379.517822265625;
float mean_s7 = 314.7926330566406;
float mean_s8 = 378.5710754394531;
float mean_x = -0.16732923686504364;
float mean_y = -0.14998193085193634;
float mean_z = 2.568124294281006;

float std_s1 = 75.8375015258789;
float std_s2 = 72.86641693115234; 
float std_s3 = 77.235107421875; 
float std_s4 = 79.19815826416016; 
float std_s5 = 76.76965332031255; 
float std_s6 = 80.93759155273438; 
float std_s7 = 74.02841949462899; 
float std_s8 = 88.13593292236328;
float std_x = 8.387557983398438;
float std_y = 8.411337852478027;
float std_z = 4.372629642486572;

// Trimmed Dataset Statistics
// float mean_s1 = 440.56646728515625;
// float mean_s2 = 369.13848876953125;
// float mean_s3 = 411.2698059082031;
// float mean_s4 = 370.88800048828125;
// float mean_s5 = 349.4165954589844;
// float mean_s6 = 398.39044189453125;
// float mean_s7 = 339.6324768066406;
// float mean_s8 = 392.6517639160156;
// float mean_x = 0.2750225365161896;
// float mean_y = 0.10459873825311661;
// float mean_z = 4.4806132316589355;

// float std_s1 = 69.91178894042969;
// float std_s2 = 66.89142608642578; 
// float std_s3 = 72.11554718017578; 
// float std_s4 = 72.93818664550781; 
// float std_s5 = 71.96635437011719; 
// float std_s6 = 72.08090209960938; 
// float std_s7 = 68.62278747558594; 
// float std_s8 = 79.95512390136719;
// float std_x = 8.54604721069336;
// float std_y = 8.562744140625;
// float std_z = 5.032437324523926;

// ------------------------------------------------------------------
// define variables for coils, PWM1 - Coil1 etc.
int pwm1 = 6;    //PWM for Coil 1
int pwm2 = 5;    //PWM for Coil 2
int pwm3 = 4;    //PWM for Coil 3
// int pwm4 = 3;    //PWM for Coil 4
int pwm5 = 2;    //PWM for Coil 5  
int pwm6 = 1;    //PWM for Coil 6
int pwm7 = 0;    //PWM for Coil 7

//define analogue ports for reading sensor values
int Sen1 = A0;      //Sensor 1 Analogue Input
int Sen2 = A1;      //Sensor 2 Analogue Input
int Sen3 = A2;      //Sensor 3 Analogue Input
int Sen4 = A3;      //Sensor 4 Analogue Input
int Sen5 = A4;      //Sensor 5 Analogue Input
int Sen6 = A5;      //Sensor 6 Analogue Input
int Sen7 = A6;      //Sensor 7 Analogue Input
int Sen8 = A7;      //Sensor 8 Analogue Input

// define variables to hold sensor values
int s_val1 = 0;     //Variables for storing Analogue values
int s_val2 = 0;
int s_val3 = 0;
int s_val4 = 0;
int s_val5 = 0;
int s_val6 = 0;
int s_val7 = 0;
int s_val8 = 0;

// define variables to hold filtered values
int est1 = 0;
int est2 = 0;
int est3 = 0;
int est4 = 0;
int est5 = 0;
int est6 = 0;
int est7 = 0;
int est8 = 0;

// define variables to hold normalised values
float n_est1 = 0;
float n_est2 = 0;
float n_est3 = 0;
float n_est4 = 0;
float n_est5 = 0;
float n_est6 = 0;
float n_est7 = 0;
float n_est8 = 0;

// ------------------------------------------------------------------
// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
constexpr int kTensorArenaSize = 70*1024;
uint8_t tensor_arena[kTensorArenaSize];
int inference_count = 0;
}  // namespace

// ------------------------------------------------------------------
// Initialize matrix transfrom input/output data
double F_x = 0;
double F_y = 0;
double F_z = 0.92;
double T_x = 0;
double T_y = 0;
double T_z = 0;
double I[9];

// define variables to hold current values for each coil
double I1;
double I2;
double I3;
double I4;
double I5;
double I6;
double I7;
double I8;
double I9;

// ------------------------------------------------------------------
// PID Initialization
// Define PID input, output, and setpoint variables
double setPoint, force_input, force_output;
// Specify the PID tuning parameters (P, I, D)
double Kp = 2.0, Ki = 0.0, Kd = 1.0;
// Gravity feedforward force
double G = 9.8; 
// Initialize the PID controller
PID myPID(&force_input, &force_output, &setPoint, Kp, Ki, Kd, DIRECT);


void setup() {
  Serial.begin(9600);

  myPID.SetMode(AUTOMATIC);
  // Set the setpoint of the PID controller
  setPoint = 100;

  // Set PWM pins
  pinMode(PH_6, OUTPUT);    // output pin for pwm4
  pinMode(PJ_7, OUTPUT);    // output pin for pwm8
  pinMode(PJ_10, OUTPUT);   // output pin for pwm9

  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  model = tflite::GetModel(position_model_no_quant);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter, "Model version mismatch!");
    return;
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Failed to allocate tensors!");
    return;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  inference_count = 0;
}

void loop() {
  if (Serial.available() > 0) {
    // Read the incoming data as a string until newline
    String data = Serial.readStringUntil('\n');
    // Parse the data into individual sensor values
    int values[8]; // Array to hold sensor values
    parseCSVData(data, values); // Implement this to parse the CSV string into values

    s_val1 = analogRead(Sen1);        // Raw Sensor Readings
    s_val2 = analogRead(Sen2);
    s_val3 = analogRead(Sen3);
    s_val4 = analogRead(Sen4);
    s_val5 = analogRead(Sen5);
    s_val6 = analogRead(Sen6);
    s_val7 = analogRead(Sen7);
    s_val8 = analogRead(Sen8);

    unsigned long startTime = micros();  // Start timing the inference
    // Keep filtering section to provide better analysis including 
    // timing considerations 
    // Filtered Sensor Readings
    est1 = filter1.updateEstimate(s_val1);
    est2 = filter2.updateEstimate(s_val2);
    est3 = filter3.updateEstimate(s_val3);
    est4 = filter4.updateEstimate(s_val4);
    est5 = filter5.updateEstimate(s_val5);
    est6 = filter6.updateEstimate(s_val6);
    est7 = filter7.updateEstimate(s_val7);
    est8 = filter8.updateEstimate(s_val8);

    // Access sensor values for the current row
    n_est1 = normalize(values[0], mean_s1, std_s1);
    n_est2 = normalize(values[1], mean_s2, std_s2);
    n_est3 = normalize(values[2], mean_s3, std_s3);
    n_est4 = normalize(values[3], mean_s4, std_s4);
    n_est5 = normalize(values[4], mean_s5, std_s5);
    n_est6 = normalize(values[5], mean_s6, std_s6);
    n_est7 = normalize(values[6], mean_s7, std_s7);
    n_est8 = normalize(values[7], mean_s8, std_s8);

    float* input_data = interpreter->typed_input_tensor<float>(0);
    input_data[0] = n_est1;
    input_data[1] = n_est2;
    input_data[2] = n_est3;
    input_data[3] = n_est4;
    input_data[4] = n_est5;
    input_data[5] = n_est6;
    input_data[6] = n_est7;
    input_data[7] = n_est8;

    // Run inference, and report any error
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed on x: %f\n",
                          static_cast<double>(n_est1));
      return;
    }

    // Get the output directly from the output tensor
    float* output_data = interpreter->typed_output_tensor<float>(0);

    // No need for dequantization
    float x = output_data[0];
    float y = output_data[1];
    float z = output_data[2];

    // Denormalization if needed
    float x_denorm = denormalize(x, mean_x, std_x);
    float y_denorm = denormalize(y, mean_y, std_y);
    float z_denorm = denormalize(z, mean_z, std_z);

    inference_count += 1;
    if (inference_count >= kInferencesPerCycle) inference_count = 0;

    // Dummy PID for timing purposes
    myPID.Compute();

    matrix_transform(x_denorm, y_denorm, z_denorm, F_x, F_y, F_z, T_x, T_y, I);

    // scale up output currents and constrain it between 0 and 255
    I1 = constrain(round(I[0] * 350),0,255);
    I2 = constrain(round(I[1] * 350),0,255);
    I3 = constrain(round(I[2] * 350),0,255);
    I4 = constrain(round(I[3] * 350),0,255);
    I5 = constrain(round(I[4] * 350),0,255);
    I6 = constrain(round(I[5] * 350),0,255);
    I7 = constrain(round(I[6] * 350),0,255);
    I8 = constrain(round(I[7] * 350),0,255);
    I9 = constrain(round(I[8] * 350),0,255);

    // assign scaled ouput values to pwm ports
    analogWrite(pwm1, I1);
    analogWrite(pwm2, I2);
    analogWrite(pwm3, I3);
    //analogWrite(PH_6, I4);
    analogWrite(pwm5, I5);
    analogWrite(pwm6, I6);
    analogWrite(pwm7, I7);
    //analogWrite(PJ_7, I8);
    //analogWrite(PJ_10, I9);

    unsigned long endTime = micros();
    unsigned long executionTime = endTime - startTime;

    Serial.println(String(x_denorm) + "," + String(y_denorm) + "," + String(z_denorm) + "," + executionTime);
  }
}

// ------------------------------------------------------------
// declare functions
// ------------------------------------------------------------
void parseCSVData(String data, int values[]) {
  int lastIndex = 0, nextIndex;
  for (int i = 0; i < 8; i++) {
    nextIndex = data.indexOf(',', lastIndex);
    if (nextIndex == -1) nextIndex = data.length();
    values[i] = data.substring(lastIndex, nextIndex).toInt();
    lastIndex = nextIndex + 1;
  }
}
float normalize(int input, float mean, float std){      // Normalizing Input Values
  float normed;
  normed = (input-mean)/std;
  return normed;
}

float denormalize(float output, float mean, float std ){   //Denormalizing Output values
  float denormed;
  denormed = (output*std) + mean;
  return denormed;
}
