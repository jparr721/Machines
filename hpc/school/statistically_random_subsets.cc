#include <cstdlib>
#include <cmath>
#include "statistically_random_subsets.h"

namespace stats {
  std::vector<int> sort(std::vector<int> unsorted_vector) {

  }

  std::vector<int> StatisticallyRandomSubsets::generate(int k, std::vector<int> n) {
    int vector_length = n.size();

    for (int i = 0; i < k; ++i) {
      const int width = vector_length -i;
      const int random_index = width * rand() % k;

      int key = n[i];
      int swap = n[random_index + i];

      n[random_index] = key;
      n[i] = swap;
    }

    // If we don't care about preserving the list
    // n.resize(50);

    std::vector<int> random_list();

    // Load the data into the vector
    std::vector<int> sub(&n[0], &n[50]);

    return random_list();
  }
} // namespace stats
