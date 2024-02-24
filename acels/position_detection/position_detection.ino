/*
 * Working Version with Normalization.
 * Not quantized model with 0.03mm MAE in normalized mode.
 */


#include <SimpleKalmanFilter.h>
#include <TensorFlowLite.h>
#include "main_functions.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "constants.h"
#include "model.h"
#include "output_handler.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

SimpleKalmanFilter filter1(5, 5, 0.01);
SimpleKalmanFilter filter2(5, 5, 0.01);
SimpleKalmanFilter filter3(5, 5, 0.01);
SimpleKalmanFilter filter4(5, 5, 0.01);
SimpleKalmanFilter filter5(5, 5, 0.01);
SimpleKalmanFilter filter6(5, 5, 0.01);
SimpleKalmanFilter filter7(5, 5, 0.01);
SimpleKalmanFilter filter8(5, 5, 0.01);

//unsigned long int milli_time;
float mean_s1 = 424.183563;
float mean_s2 = 342.420349;
float mean_s3 = 393.998260;
float mean_s4 = 342.354767;
float mean_s5 = 322.610107;
float mean_s6 = 372.975006;
float mean_s7 = 306.397583;
float mean_s8 = 372.363190;
float mean_x = -0.802820;
float mean_y = 0.090302;
float mean_z = 3.273132;


float std_s1 = 88.877754;
float std_s2 = 84.047852; 
float std_s3 = 90.260513; 
float std_s4 = 90.475922; 
float std_s5 = 87.819214; 
float std_s6 = 93.413918; 
float std_s7 = 83.542412; 
float std_s8 = 101.125404;
float std_x = 9.355948;
float std_y = 9.360116;
float std_z = 5.091211;


int pwm1 = 6;       //PWM for Coil 1
int pwm2 = 5;       //PWM for Coil 2
int pwm3 = 4;       //PWM for Coil 3
int pwm4 = 3;       //PWM for Coil 4
int pwm5 = 2;       //PWM for Coil 5  
int pwm6 = 1;       //PWM for Coil 6
int pwm7 = 0;       //PWM for Coil 7

int Sen1 = A0;      //Sensor 1 Analogue Input
int Sen2 = A1;      //Sensor 2 Analogue Input
int Sen3 = A2;      //Sensor 3 Analogue Input
int Sen4 = A3;      //Sensor 4 Analogue Input
int Sen5 = A4;      //Sensor 5 Analogue Input
int Sen6 = A5;      //Sensor 6 Analogue Input
int Sen7 = A6;      //Sensor 7 Analogue Input
int Sen8 = A7;      //Sensor 8 Analogue Input

int s_val1 = 0;     //Variables for storing Analogue values
int s_val2 = 0;
int s_val3 = 0;
int s_val4 = 0;
int s_val5 = 0;
int s_val6 = 0;
int s_val7 = 0;
int s_val8 = 0;

int est1 = 0;
int est2 = 0;
int est3 = 0;
int est4 = 0;
int est5 = 0;
int est6 = 0;
int est7 = 0;
int est8 = 0;

float n_est1 = 0;
float n_est2 = 0;
float n_est3 = 0;
float n_est4 = 0;
float n_est5 = 0;
float n_est6 = 0;
float n_est7 = 0;
float n_est8 = 0;


// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count = 0;

constexpr int kTensorArenaSize = 70*1024;
uint8_t tensor_arena[kTensorArenaSize];
}  // namespace


void setup() {

  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.

  extern unsigned char model_xyz_quantized[];
  
  model = tflite::GetModel(model_xyz_quantized);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // This pulls in all the operation implementations we need.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::AllOpsResolver resolver;

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  
  output = interpreter->output(0);

  // Keep track of how many inferences we have performed.
  inference_count = 0;
}

// The name of this function is important for Arduino compatibility.
void loop() {

  unsigned long startTime = micros();

  s_val1 = analogRead(Sen1);        // Raw Sensor Readings
  s_val2 = analogRead(Sen2);
  s_val3 = analogRead(Sen3);
  s_val4 = analogRead(Sen4);
  s_val5 = analogRead(Sen5);
  s_val6 = analogRead(Sen6);
  s_val7 = analogRead(Sen7);
  s_val8 = analogRead(Sen8);

  est1 = filter1.updateEstimate(s_val1);    // Filtered Sensor Readings
  est2 = filter2.updateEstimate(s_val2);
  est3 = filter3.updateEstimate(s_val3);
  est4 = filter4.updateEstimate(s_val4);
  est5 = filter5.updateEstimate(s_val5);
  est6 = filter6.updateEstimate(s_val6);
  est7 = filter7.updateEstimate(s_val7);
  est8 = filter8.updateEstimate(s_val8);


  n_est1 = normalize(est1, mean_s1, std_s1);    // NN Input Values
  n_est2 = normalize(est2, mean_s2, std_s2);
  n_est3 = normalize(est3, mean_s3, std_s3);
  n_est4 = normalize(est4, mean_s4, std_s4);
  n_est5 = normalize(est5, mean_s5, std_s5);
  n_est6 = normalize(est6, mean_s6, std_s6);
  n_est7 = normalize(est7, mean_s7, std_s7);
  n_est8 = normalize(est8, mean_s8, std_s8);

  // Quantize the input from floating-point to integer
  int8_t s1_quantized = n_est1 / input->params.scale + input->params.zero_point;
  int8_t s2_quantized = n_est2 / input->params.scale + input->params.zero_point;
  int8_t s3_quantized = n_est3 / input->params.scale + input->params.zero_point;
  int8_t s4_quantized = n_est4 / input->params.scale + input->params.zero_point;
  int8_t s5_quantized = n_est5 / input->params.scale + input->params.zero_point;
  int8_t s6_quantized = n_est6 / input->params.scale + input->params.zero_point;
  int8_t s7_quantized = n_est7 / input->params.scale + input->params.zero_point;
  int8_t s8_quantized = n_est8 / input->params.scale + input->params.zero_point;

  input->data.int8[0] = s1_quantized;
  input->data.int8[1] = s2_quantized;
  input->data.int8[2] = s3_quantized;
  input->data.int8[3] = s4_quantized;
  input->data.int8[4] = s5_quantized;
  input->data.int8[5] = s6_quantized;
  input->data.int8[6] = s7_quantized;
  input->data.int8[7] = s8_quantized;

  // Run inference, and report any error
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed on x: %f\n",
                         static_cast<double>(n_est1));
    return;
  }

  // Obtain the quantized output from model's output tensor
  int8_t x_quantized = output->data.int8[0];
  int8_t y_quantized = output->data.int8[1];
  int8_t z_quantized = output->data.int8[2];

  float x = (x_quantized - output->params.zero_point) * output->params.scale;
  float y = (y_quantized - output->params.zero_point) * output->params.scale;
  float z = (z_quantized - output->params.zero_point) * output->params.scale;


  float x_denorm = denormalize(x, mean_x, std_x);
  float y_denorm = denormalize(y, mean_y, std_y);
  float z_denorm = denormalize(z, mean_z, std_z);



  // Output the results. A custom HandleOutput function can be implemented
  // for each supported hardware target.
  //HandleOutput(error_reporter, x, y, z, roll, pitch);


  // Serial.print(x_denorm, 3);
  // Serial.print(", ");
  // Serial.print(y_denorm, 3);
  // Serial.print(", ");
  // Serial.print(z_denorm, 3);

  // Serial.println();
  // delay(200);

  // Increment the inference_counter, and reset it if we have reached
  // the total number per cycle
  inference_count += 1;
  if (inference_count >= kInferencesPerCycle) inference_count = 0;

  unsigned long endTime = micros();
  unsigned long executionTime = endTime - startTime;
  Serial.print("Execution time: ");
  Serial.print(executionTime);
  Serial.println(" us");
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