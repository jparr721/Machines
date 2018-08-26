#include <iostream>
#include <stdio.h>
#include <omp.h>
#include <vector>

int private_threads(void) {
  omp_set_num_threads(8);
  int num_threads, thread_id;

  #pragma omp parallel private(num_threads, thread_id)
  {
    thread_id = omp_get_thread_num();
    std::cout << "thread number: " << thread_id << std::endl;
    if (thread_id == 0) {
      num_threads = omp_get_num_threads();
      std::cout << "Total threads: " << num_threads << std::endl;
    }
  }

  return 0;
}

int parallel_loop(void) {
  const int N = 20;
  int nthreads, threadid;

  std::vector<double> a, b, result;

  // Initialize
  for (int i = 0; i < N; i++){
    a.push_back(1.0 * i);
    b.push_back(2.0 * i);
  }

#pragma omp parallel private(threadid)
  {
  threadid = omp_get_thread_num();

  #pragma omp for
  for (int i = 0; i < N; i++) {
    std::cout << "a[i]: " << a[i] << std::endl;
    std::cout << "b[i]: " << b[i] << std::endl;
    std::cout << a[i] + b[i] << std::endl;
    result.push_back(a[i] + b[i]);
    std::cout << "Thread id: " << threadid << "working on index: " << i << std::endl;
  }
  }

  std::cout << result.size() << std::endl;

  for (std::vector<double>::size_type i = 0; i != result.size(); ++i) {
    std::cout << i << ": " << result[i] << std::endl;
  }
  std::cout << "Test result[19] = " << result[19] << std::endl;

  return 0;
}

int main(void) {
  parallel_loop();
  /* private_threads(); */

  return 0;
}
