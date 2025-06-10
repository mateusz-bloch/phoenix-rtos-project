#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/time.h>

#include "model_data.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"

constexpr int kTensorArenaSize = 40 * 1024;  // 32 KB – dostosuj wg potrzeb
static uint8_t tensor_arena[kTensorArenaSize];

static uint64_t now_us(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (uint64_t)tv.tv_sec * 1000000ull + tv.tv_usec;
}

//    Zakładamy model float32 o kształcie [1, N, M, C]
void FillInput(TfLiteTensor* input, bool gesture_active) {
  const float v = gesture_active ? 1.0f : 0.0f;
  const int count = input->bytes / sizeof(float);
  for (int i = 0; i < count; ++i) {
    input->data.f[i] = v;
  }
}

int cmp_uint64(const void* a, const void* b) {
  uint64_t ua = *(const uint64_t*)a;
  uint64_t ub = *(const uint64_t*)b;
  return (ua > ub) - (ua < ub);
}

int main(void) {
  const tflite::Model* model = tflite::GetModel(g_magic_wand_model_data);
  printf("Model size (FLASH): %d bytes\n", g_magic_wand_model_data_len);

  constexpr int kOpCount = 7;
  static tflite::MicroMutableOpResolver<kOpCount> resolver;
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddReshape();
  resolver.AddAveragePool2D();
  resolver.AddMean();
  

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
    times_us[i] = now_us() - t0;
  }

  qsort(times_us, kIters, sizeof(uint64_t), cmp_uint64);

  uint64_t sum = 0;
  for (int i = 0; i < kIters; ++i)
    sum += times_us[i];

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
  printf("Arena total: %d bytes (użycie nieznane)\n", kTensorArenaSize);
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
