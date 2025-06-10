#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/time.h>
#include "model_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"

constexpr int kTensorArenaSize = 320 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];

void FillInput(TfLiteTensor* input, bool person_present) {
  uint8_t v = person_present ? 255 : 0;
  for (int i = 0; i < input->bytes; i++) {
    input->data.uint8[i] = v;
  }
}

static uint64_t now_us(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (uint64_t)tv.tv_sec * 1000000ull + tv.tv_usec;
}

int cmp_uint64(const void* a, const void* b) {
  uint64_t ua = *(const uint64_t*)a;
  uint64_t ub = *(const uint64_t*)b;
  return (ua > ub) - (ua < ub);
}


int main(void) {
  const tflite::Model* model = tflite::GetModel(person_detection_model_data);
  printf("Model size (FLASH): %d bytes\n", person_detection_model_data_len);

  constexpr int kOpCount = 6;
  static tflite::MicroMutableOpResolver<kOpCount> resolver;
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddReshape();
  resolver.AddAveragePool2D();

  tflite::MicroInterpreter interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, nullptr);
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    printf("AllocateTensors() failed\n");
    return -1;
  }

  TfLiteTensor* input = interpreter.input(0);

  FillInput(input, false);
  interpreter.Invoke();

  const int kIters = 1000;
  uint64_t times_us[kIters];
  for (int i = 0; i < kIters; ++i) {
    FillInput(input, (i % 2) == 0);
    uint64_t t0 = now_us();
    if (interpreter.Invoke() != kTfLiteOk) {
      printf("Invoke() failed at iter %d\n", i);
      return -1;
    }
    uint64_t t1 = now_us();
    times_us[i] = t1 - t0;
  }

  qsort(times_us, kIters, sizeof(uint64_t), cmp_uint64);

  uint64_t sum = 0, mn = UINT64_MAX, mx = 0;
  for (int i = 0; i < kIters; ++i) {
    uint64_t t = times_us[i];
    sum += t;
    if (t < mn) mn = t;
    if (t > mx) mx = t;
  }
  double avg = sum / (double)kIters;
  double median = (kIters % 2 == 0) ?
    (times_us[kIters / 2 - 1] + times_us[kIters / 2]) / 2.0 :
    times_us[kIters / 2];
  double p95 = times_us[(int)(0.95 * kIters)];
  uint64_t min = times_us[0];
  uint64_t max = times_us[kIters - 1];

  printf("---- Wyniki inferencji [%d iteracji] ----\n", kIters);
  printf("Avg latency: %.3f ms\n", avg / 1000.0);
  printf("Median latency: %.3f ms\n", median / 1000.0);
  printf("p95 latency: %.3f ms\n", p95 / 1000.0);
  printf("Min latency: %.3f ms\n", min / 1000.0);
  printf("Max latency: %.3f ms\n", max / 1000.0);
  printf("Jitter (max - min): %.3f ms\n", (max - min) / 1000.0);

#if defined(TFLITE_MICRO_USE_CONSTRAINED_HEAP)
  printf("Arena total: %d bytes (użycie nieznane w tej wersji TFLM)\n",
         kTensorArenaSize);
#else
  size_t used = kTensorArenaSize - interpreter.arena_used_bytes();
  printf("Arena total: %d bytes; Used: %zu bytes\n",
         kTensorArenaSize, used);
#endif

  for (int i = 0; i < kIters; ++i) {
    printf("%d,%llu\n", i, times_us[i]);
  }

  return 0;
}
