#include <iostream>
#include <random>
#include <vector>

std::vector<std::pair<int, int>> train_data = {
    {0, 0}, {1, 2}, {2, 4}, {3, 6}, {4, 8}};

const size_t data_size = train_data.size();

float rand_float(float a, float b) {
  static std::random_device rd;
  static std::mt19937 gen(101);
  std::uniform_real_distribution<float> dist(a, b);
  return dist(gen);
}

float loss(float m, float bias) {
  float error = 0.f;
  for (auto &p : train_data) {
    float x = p.first;
    float y = m * x + bias;
    float d = y - p.second;
    error += d * d; // mse
  }

  return error /= data_size; // average
}

int main(void) {
  // y = m*x + c
  float m = rand_float(0, 10);
  float bias = rand_float(0, 5);

  float eps = 1e-3;
  float learning_rate = 1e-3;

  std::cout << loss(m, bias) << std::endl;
  for (int i = 0; i < 500; i++) {
    float l = loss(m, bias);
    float d_m = (loss(m + eps, bias) - l) / eps; // finite difference
    float d_bias = (loss(m, bias + eps) - l) / eps;

    m -= learning_rate * d_m;
    bias -= learning_rate * d_bias;

    std::cout << "loss: " << loss(m, bias) << " " << "m: " << m << " "
              << "b: " << bias << std::endl;
  }

  std::cout << "-------------------------------" << std::endl;
  std::cout << "m after training = : " << m << std::endl;

  return 0;
}
