#include <array>
#include <iostream>
#include <random>

// AND gate
std::array<std::array<int, 3>, 4> train_data = {
    {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 1}}};

const size_t data_size = train_data.size();

float rand_float(float a, float b) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(a, b);
  return dist(gen);
}

float sigmoidf(float x) { return 1 / (1 + expf(-x)); }

float loss(float m1, float m2, float bias) {
  float result = 0.f;
  for (auto &p : train_data) {
    float x1 = p[0];
    float x2 = p[1];
    float y = sigmoidf(x1 * m1 + x2 * m2 + bias);
    float error = y - p[2];
    result += error * error;
  }
  return result / data_size;
}

int main(void) {
  float m1 = rand_float(0, 1);
  float m2 = rand_float(0, 1);
  float bias = rand_float(0, 1);

  float eps = 1e-1;
  float learning_rate = 1e-1;

  for (int i = 0; i < 1000 * 1000; i++) {
    float l = loss(m1, m2, bias);
    // std::cout << l << std::endl;
    float d_m1 = (loss(m1 + eps, m2, bias) - l) / eps;
    float d_m2 = (loss(m1, m2 + eps, bias) - l) / eps;
    float d_bias = (loss(m1, m2, bias + eps) - l) / eps;
    m1 -= learning_rate * d_m1;
    m2 -= learning_rate * d_m2;
    bias -= learning_rate * d_bias;
  }

  // std::cout << "m1: " << m1 << " " << "m2: " << m2
  //           << " "
  //              "b: "
  //           << bias << "loss: " << loss(m1, m2, bias) << std::endl;

  std::cout << "Output: " << std::endl;

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      std::cout << i << " | " << j << " = " << sigmoidf(i * m1 + j * m2 + bias)
                << std::endl;
    }
  }

  return 0;
}
