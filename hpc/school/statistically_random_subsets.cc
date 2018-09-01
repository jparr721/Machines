#include "statistically_random_subsets.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>

namespace stats {
std::vector<int> StatisticallyRandomSubsets::generate(int k, const std::vector<int> & n) {
  std::vector<int> random_list(n);
  std::cout << &random_list[k] << std::endl;
  std::random_shuffle(&random_list[0], &random_list[k]);

  /* for (const auto & val : random_list) { */
  /*   std::cout << val << std::endl; */
  /* } */

  /* int vector_length = n.size(); */

  /* std::default_random_engine generator; */
  /* std::uniform_int_distribution<int> distribution(0, vector_length); */

  /* int width; */
  /* int random_index; */

  /* for (int i = 0; i < k; ++i) { */
  /*   // Random number from 0 to n.size() */
  /*   random_index = distribution(generator); */
  /*   width = vector_length - i; */

  /*   int key = n[i]; */
  /*   int swap = n[random_index + i]; */

  /*   n[random_index] = key; */
  /*   n[i] = swap; */
  /* } */

  /* // If we don't care about preserving the list */
  /* // n.resize(50); */

  /* std::vector<int> random_list(); */

  /* // Load the data into the vector */
  /* std::vector<int> sub(&n[0], &n[50]); */

  return random_list;
}

} // namespace stats

int main() {
  stats::StatisticallyRandomSubsets srs;

  std::vector<int> n;
  n.reserve(500);
  int k = 50;
  for (int i = 0; i < 500; ++i) {
    n.push_back(i);
  }

  std::vector<int> output = srs.generate(k, n);

  for (const auto & val : output) {
    std::cout << val << std::endl;
  }

  return 0;
}
