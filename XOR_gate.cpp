#include <array>
#include <iostream>
#include <random>

// xor data
std::array<std::array<int, 3>, 4> train_data = {{
    {0, 0, 0},
    {1, 0, 1},
    {0, 1, 1},
    {1, 1, 0},
}};

const size_t data_size = train_data.size();

struct Xor {
  float or_m1;
  float or_m2;
  float or_b;

  float nand_m1;
  float nand_m2;
  float nand_b;

  float and_m1;
  float and_m2;
  float and_b;
};

const float eps = 1e-1;

float rand_float() {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::uniform_real_distribution<float> dist(0.f, 1.f);
  return dist(gen);
}

float sigmoidf(float x) { return 1 / (1 + expf(-x)); }

float forward(Xor &m, float x1, float x2) {
  float a = sigmoidf(m.or_m1 * x1 + m.or_m2 * x2 + m.or_b);
  float b = sigmoidf(m.nand_m1 * x1 + m.nand_m2 * x2 + m.nand_b);
  return sigmoidf(m.and_m1 * a + m.and_m2 * b + m.and_b);
}

float loss(Xor &m) {
  float result = 0.f;
  for (auto &p : train_data) {
    float x1 = p[0];
    float x2 = p[1];
    float y = forward(m, x1, x2);
    float error = y - p[2];
    result += error * error;
  }
  return result / data_size;
}

Xor rand_xor(void) {
  Xor m;
  m.or_m1 = rand_float();
  m.or_m2 = rand_float();
  m.or_b = rand_float();
  m.nand_m1 = rand_float();
  m.nand_m2 = rand_float();
  m.nand_b = rand_float();
  m.and_m1 = rand_float();
  m.and_m2 = rand_float();
  m.and_b = rand_float();
  return m;
}

void print_xor(const Xor &m) {
  std::cout << "m.or_m1 = " << m.or_m1 << std::endl;
  std::cout << "m.or_m2 = " << m.or_m2 << std::endl;
  std::cout << "m.or_b = " << m.or_b << std::endl;
  std::cout << "m.nand_m1 = " << m.nand_m1 << std::endl;
  std::cout << "m.nand_m2 = " << m.nand_m2 << std::endl;
  std::cout << "m.nand_b = " << m.nand_b << std::endl;
  std::cout << "m.and_m1 = " << m.and_m1 << std::endl;
  std::cout << "m.and_m2 = " << m.and_m2 << std::endl;
  std::cout << "m.and_b = " << m.and_b << std::endl;
}

Xor finite_difference(Xor &m) {
  Xor g;
  float l = loss(m);
  float saved;

  saved = m.or_m1;
  m.or_m1 += eps;
  g.or_m1 = (loss(m) - l) / eps;
  m.or_m1 = saved;

  saved = m.or_m2;
  m.or_m2 += eps;
  g.or_m2 = (loss(m) - l) / eps;
  m.or_m2 = saved;

  saved = m.or_b;
  m.or_b += eps;
  g.or_b = (loss(m) - l) / eps;
  m.or_b = saved;

  saved = m.nand_m1;
  m.nand_m1 += eps;
  g.nand_m1 = (loss(m) - l) / eps;
  m.nand_m1 = saved;

  saved = m.nand_m2;
  m.nand_m2 += eps;
  g.nand_m2 = (loss(m) - l) / eps;
  m.nand_m2 = saved;

  saved = m.nand_b;
  m.nand_b += eps;
  g.nand_b = (loss(m) - l) / eps;
  m.nand_b = saved;

  saved = m.and_m1;
  m.and_m1 += eps;
  g.and_m1 = (loss(m) - l) / eps;
  m.and_m1 = saved;

  saved = m.and_m2;
  m.and_m2 += eps;
  g.and_m2 = (loss(m) - l) / eps;
  m.and_m2 = saved;

  saved = m.and_b;
  m.and_b += eps;
  g.and_b = (loss(m) - l) / eps;
  m.and_b = saved;

  return g;
}

int main(void) {
  Xor m = rand_xor();

  Xor g = finite_difference(m);

  return 0;
}
