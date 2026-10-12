// gpubw.hip: GPU memory read and copy bandwidth on one device (device buffers, 1 GiB each).
#include <hip/hip_runtime.h>
#include <cstdio>
__global__ void rd(const float4* __restrict__ a, size_t n, float* out) {
  float4 s = {0, 0, 0, 0};
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    float4 v = a[i]; s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w; }
  if (s.x == 12345.f) out[0] = s.y + s.z + s.w;
}
__global__ void cp(const float4* __restrict__ a, float4* __restrict__ b, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) b[i] = a[i];
}
int main() {
  size_t bytes = 1ull << 30, n = bytes / sizeof(float4);
  float4 *a, *b; float* o;
  hipMalloc(&a, bytes); hipMalloc(&b, bytes); hipMalloc(&o, 4);
  hipMemset(a, 1, bytes); hipMemset(b, 0, bytes);
  hipEvent_t s, e; hipEventCreate(&s); hipEventCreate(&e);
  float best_rd = 0, best_cp = 0, ms;
  for (int r = 0; r < 20; r++) {
    hipEventRecord(s); rd<<<4096, 256>>>(a, n, o); hipEventRecord(e); hipEventSynchronize(e);
    hipEventElapsedTime(&ms, s, e); float g = bytes / (ms * 1e6f); if (g > best_rd) best_rd = g;
    hipEventRecord(s); cp<<<4096, 256>>>(a, b, n); hipEventRecord(e); hipEventSynchronize(e);
    hipEventElapsedTime(&ms, s, e); g = 2.f * bytes / (ms * 1e6f); if (g > best_cp) best_cp = g;
  }
  printf("gpu_read_best_GBps %.1f\ngpu_copy_best_GBps %.1f\n", best_rd, best_cp);
  return 0;
}
