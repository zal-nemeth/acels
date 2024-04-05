/*
 * Working Version with Normalization.
 */

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

// Change mean and std values based on dataset used for model training
// -----------------------------------------------
// Extended Dataset Statistics
// float mean_s1 = 431.5683288574219;
// float mean_s2 = 352.7138366699219;
// float mean_s3 = 402.7107238769531;
// float mean_s4 = 350.7831726074219;
// float mean_s5 = 331.1814270019531;
// float mean_s6 = 379.517822265625;
// float mean_s7 = 314.7926330566406;
// float mean_s8 = 378.5710754394531;
// float mean_x = -0.16732923686504364;
// float mean_y = -0.14998193085193634;
// float mean_z = 2.568124294281006;

// float std_s1 = 75.8375015258789;
// float std_s2 = 72.86641693115234; 
// float std_s3 = 77.235107421875; 
// float std_s4 = 79.19815826416016; 
// float std_s5 = 76.76965332031255; 
// float std_s6 = 80.93759155273438; 
// float std_s7 = 74.02841949462899; 
// float std_s8 = 88.13593292236328;
// float std_x = 8.387557983398438;
// float std_y = 8.411337852478027;
// float std_z = 4.372629642486572;

// -----------------------------------------------
// Trimmed Dataset Statistics
float mean_s1 = 440.56646728515625;
float mean_s2 = 369.13848876953125;
float mean_s3 = 411.2698059082031;
float mean_s4 = 370.88800048828125;
float mean_s5 = 349.4165954589844;
float mean_s6 = 398.39044189453125;
float mean_s7 = 339.6324768066406;
float mean_s8 = 392.6517639160156;
float mean_x = 0.2750225365161896;
float mean_y = 0.10459873825311661;
float mean_z = 4.4806132316589355;

float std_s1 = 69.91178894042969;
float std_s2 = 66.89142608642578; 
float std_s3 = 72.11554718017578; 
float std_s4 = 72.93818664550781; 
float std_s5 = 71.96635437011719; 
float std_s6 = 72.08090209960938; 
float std_s7 = 68.62278747558594; 
float std_s8 = 79.95512390136719;
float std_x = 8.54604721069336;
float std_y = 8.562744140625;
float std_z = 5.032437324523926;

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
constexpr int kTensorArenaSize = 70*1024;
uint8_t tensor_arena[kTensorArenaSize];
int inference_count = 0;
}  // namespace

void setup() {
  Serial.begin(9600);
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

}

void loop() {
  if (Serial.available() > 0) {
    // Read the incoming data as a string until newline
    String data = Serial.readStringUntil('\n');
    // Parse the data into individual sensor values
    int values[8]; // Array to hold sensor values
    parseCSVData(data, values); // Implement this to parse the CSV string into values

    unsigned long startTime = micros();
    // NN Input Values
    n_est1 = normalize(values[0], mean_s1, std_s1);
    n_est2 = normalize(values[1], mean_s2, std_s2);
    n_est3 = normalize(values[2], mean_s3, std_s3);
    n_est4 = normalize(values[3], mean_s4, std_s4);
    n_est5 = normalize(values[4], mean_s5, std_s5);
    n_est6 = normalize(values[5], mean_s6, std_s6);
    n_est7 = normalize(values[6], mean_s7, std_s7);
    n_est8 = normalize(values[7], mean_s8, std_s8);

    // Quantize the input from floating-point to integer
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

    // Increment the inference_counter, and reset it if we have reached
    // the total number per cycle
    inference_count += 1;
    if (inference_count >= kInferencesPerCycle) inference_count = 0;

    unsigned long endTime = micros();
    unsigned long executionTime = endTime - startTime;

    Serial.println(String(x_denorm) + "," + String(y_denorm) + "," + String(z_denorm) + "," + executionTime);
  }
}

// --------------------------------------------------
// Define Functions
// --------------------------------------------------
// Parse input data from csv file through serial monitor
void parseCSVData(String data, int values[]) {
  int lastIndex = 0, nextIndex;
  for (int i = 0; i < 8; i++) {
    nextIndex = data.indexOf(',', lastIndex);
    if (nextIndex == -1) nextIndex = data.length();
    values[i] = data.substring(lastIndex, nextIndex).toInt();
    lastIndex = nextIndex + 1;
  }
}
// Normalizing Input Values
float normalize(int input, float mean, float std){
  float normed;
  normed = (input-mean)/std;
  return normed;
}
// Denormalizing Output values
float denormalize(float output, float mean, float std ){
  float denormed;
  denormed = (output*std) + mean;
  return denormed;
}
