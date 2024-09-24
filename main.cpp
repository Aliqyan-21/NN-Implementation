#include <iostream>
#include <random>
#include <vector>

std::vector<std::pair<int, int>> train_data = {
    {0, 0}, {1, 2}, {2, 4}, {3, 6}, {4, 8}};

// y = mx + c

float rand_float(void) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::uniform_real_distribution<float> dist(0.f, 1.f);
  return dist(gen);
}

int main(void) {
  float x = rand_float();
  std::cout << x << std::endl;

  for (auto &p : train_data)
    std::cout << p.first << ", " << p.second << std::endl;

  return 0;
}
