#include "output_handler.h"

#include "Arduino.h"
#include "constants.h"

// The pin of the Arduino's built-in LED
//int led = LED_BUILTIN;

// Track whether the function has run at least once
bool initialized = false;

/*float denormalize(int output, float mean, float std ){   //Denormalizing Output values
  float denormed;
  denormed = (output-mean)/std;
  return denormed;
}*/

// Animates a dot across the screen to represent the current x and y values
void HandleOutput(tflite::ErrorReporter* error_reporter, float x_value,
                  float y_value, float z_value, float roll_value, float pitch_value) {
  // Do this only once
  if (!initialized) {
    // Set the LED pin to output
    //pinMode(led, OUTPUT);
    initialized = true;
  }

  /*float x_denorm = denormalize(x_value, mean_x, std_x);
  float y_denorm = denormalize(y_value, mean_y, std_y);
  float z_denorm = denormalize(z_value, mean_z, std_z);
  float roll_denorm = denormalize(roll_value, mean_roll, std_roll);
  float pitch_denorm = denormalize(pitch_value, mean_pitch, std_pitch);

  Serial.print(x_denorm);
  Serial.print(", ");
  Serial.print(y_denorm);
  Serial.print(", ");
  Serial.print(z_denorm);
  Serial.print(", ");
  Serial.print(roll_denorm);
  Serial.print(", ");
  Serial.println(pitch_denorm);
  delay(100);
  */
  // Log the current brightness value for display in the Arduino plotter
  //TF_LITE_REPORT_ERROR(error_reporter, "x: %f, y: %f, z: %f, roll: %f, pitch: %f\n", (x_value, y_value, z_value, roll_value, pitch_value));
}
