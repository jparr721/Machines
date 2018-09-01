#pragma once

#include <vector>

namespace stats {
class StatisticallyRandomSubsets {
public:
  std::vector<int> sort(std::vector<int> unsored_vector);
  std::vector<int> generate(int k, const std::vector<int> & n);
};
} // namespace stats
