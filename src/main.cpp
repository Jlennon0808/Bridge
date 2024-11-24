#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "model_data(1).h"

// TensorFlow Lite memory allocation
constexpr int kTensorArenaSize = 2 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

void app_main() {
    printf("Initializing TensorFlow Lite...\n");

    // Load model
    const tflite::Model* model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("Model schema version mismatch!\n");
        return;
    }

    // Create TFLite interpreter
    tflite::AllOpsResolver resolver;
    tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kTensorArenaSize);
    if (interpreter.AllocateTensors() != kTfLiteOk) {
        printf("Tensor allocation failed!\n");
        return;
    }

    // Get input and output tensors
    TfLiteTensor* input = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);

    // Simulate accelerometer data
    float simulated_data[3] = {0.5, 1.0, -0.5};
    for (int i = 0; i < 3; i++) {
        input->data.f[i] = simulated_data[i];
    }

    // Run inference
    if (interpreter.Invoke() != kTfLiteOk) {
        printf("Failed to invoke the model!\n");
        return;
    }

    // Print the output
    for (int i = 0; i < output->dims->data[0]; i++) {
        printf("Compressed Data [%d]: %.2f\n", i, output->data.f[i]);
    }

    while (1) {
        vTaskDelay(pdMS_TO_TICKS(2000));
    }
}
