# MNIST Digit Recognition CNN in C

A lightweight, memory-efficient implementation of a Convolutional Neural Network (CNN) for MNIST handwritten digit recognition, written in pure C without dependencies on external machine learning libraries.

## Features
- Complete CNN implementation with:
  - Convolutional layer
  - ReLU activation
  - Max pooling
  - Softmax classifier
- Memory management tools and tracking
- Training and evaluation on the MNIST dataset
- Batch processing support
- Cross-platform compatibility (Windows and UNIX-like systems)

## Requirements
- C compiler (GCC, Clang, MSVC, etc.)
- MNIST dataset files:
  - `train-images.idx3-ubyte`: Training images (60,000 samples)
  - `train-labels.idx1-ubyte`: Training labels
  - `t10k-images.idx3-ubyte`: Test images (10,000 samples)
  - `t10k-labels.idx1-ubyte`: Test labels

## Dataset
This project uses the MNIST dataset, which consists of 28x28 grayscale images of handwritten digits (0-9). The dataset is not included in this repository due to size constraints but can be downloaded from:

[MNIST Dataset Official Site](http://yann.lecun.com/exdb/mnist/)

Make sure to place all four MNIST files in the same directory as the executable.

## Building and Running
### On Linux/macOS:
```sh
gcc -o cnn CNN.c -lm
./cnn
```

### On Windows:
```sh
gcc -o cnn.exe CNN.c
cnn.exe
```

## Network Architecture
- **Input:** 28x28 grayscale images
- **Convolutional layer:** 8 filters of size 3x3
- **ReLU activation**
- **Max pooling:** 2x2 pooling
- **Softmax output layer:** 10 classes (digits 0-9)

## Implementation Details
- The CNN is implemented from scratch using only standard C libraries.
- The code includes comprehensive memory management utilities.
- Both standard and batch processing training methods are supported.
- Performance metrics (accuracy, loss) are reported during training and evaluation.
- Cross-platform compatibility with Windows and UNIX-like systems.

## Performance
With the default configuration (`5 epochs`, `learning rate 0.01`, `batch size 32`), the network typically achieves around **95% accuracy** on the MNIST test set.

## Memory Usage
The implementation includes memory tracking functionality to monitor usage:
- Current memory usage
- Peak memory usage
- Allocation and deallocation tracking
- Memory leak detection

## Configuration
The following parameters can be adjusted in the code:
```c
#define IMAGE_SIZE 28
#define FILTER_SIZE 3
#define POOL_SIZE 2
#define NUM_FILTERS 8
#define NUM_CLASSES 10
#define TEST_SAMPLES 1000
#define TRAIN_SAMPLES 60000
#define EPOCHS 5
#define LEARNING_RATE 0.01
#define BATCH_SIZE 32
```

## License
This project is open-source and available under the MIT License.

## Author
Developed by Sai Aditya Patil

