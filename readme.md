# Logic Gates trained with NN

simple implementation of neural network or perceptron in cpp. To learn about them how they work under the hood and to learn how to implement them in cpp.

#### Some Terms

- ##### Perceptron - A single neuron

- ##### NN - Neural Network

## Code Files:

- `first_perceptron.cpp:` A simple perceptron model to learn basic neural network concepts.
- `AND_gate.cpp:` Implements the AND gate using a single-layer perceptron.
- `OR_gate.cpp:` Implements the OR gate using a single-layer perceptron.
- `XOR_gate.cpp:` Implements the XOR gate, which is constructed using a combination of AND, OR, and NAND gates in a multi-layer perceptron,
  because it is not possible to contruct XOR with single neuron.

## Getting Started

### Prerequisites

- C++ compiler (GCC/Clang recommended)

### Compilation

To compile the logic gate implementations, use:

```bash
g++ AND_gate.cpp -o and_gate
g++ OR_gate.cpp -o or_gate
g++ XOR_gate.cpp -o xor_gate
g++ first_perceptron.cpp -o perceptron
```

## Notes

- The logic gates are trained using a simple gradient descent method to minimize the
  error between predicted and expected outputs.
- The loss function used is MSE (Mean Squared Error).
