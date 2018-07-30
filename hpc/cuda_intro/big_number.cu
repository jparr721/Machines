#include <iostream>
#include <math.h>


__global__
void add(int n, float* x, float* y) {
  for (int i = 0; i < n; i++)
    y[i] = x[i] + y[i];
}

int main(void) {
  int N = 1 << 20; // Bit shift up to 1m elements
  float *x, *y;

  //Allocate unified memory
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float)):

  /* float* x = new float[N]; */
  /* float* y = new float[N]; */

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  /* add(N, x, y); */
  add <<<1, 1>>>(N, x, y);

  // Wait for GPU to finish
  cudaDeviceSynchronize();

  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = std::fmax(maxError, std::fabs(y[i] - 3.0f));

  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  cudaFree(x);
  cudaFree(y);
  /* delete [] x; */
  /* delete [] y; */

  return 0;
}
