#include <iostream>
#include <random>
#include <vector>

std::vector<std::pair<int, int>> train_data = {
    {0, 0}, {1, 2}, {2, 4}, {3, 6}, {4, 8}};

const size_t data_size = train_data.size();

float rand_float(void) {
  static std::random_device rd;
  static std::mt19937 gen(101);
  static std::uniform_real_distribution<float> dist(0.f, 10.f);
  return dist(gen);
}

float loss(float m) {
  float error = 0.f;
  for (auto &p : train_data) {
    float x = p.first;
    float y = m * x;
    float d = y - p.second;
    error += d * d; // mse
  }
  error /= data_size; // average

  return error;
}

int main(void) {
  // y = m*x + c
  float m = rand_float();

  float eps = 1e-3;
  float learning_rate = 1e-3;

  std::cout << loss(m) << std::endl;
  for (int i = 0; i < 500; i++) {
    float dloss = (loss(m + eps) - loss(m)) / eps;
    m -= learning_rate * dloss;
    std::cout << "loss: " << loss(m) << std::endl;
  }

  std::cout << "-------------------------------" << std::endl;
  std::cout << "m after training = : " << m << std::endl;

  return 0;
}
