#pragma once

#include <vector>

namespace stats {
class StatisticallyRandomSubsets {
  std::vector<int> sort(std::vector<int> unsored_vector);
  std::vector<int> generate(int k, std::vector<int> n);
};
} // namespace stats
