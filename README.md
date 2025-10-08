# Artificial Neural Network from Scratch

This project implements a 2-layer Artificial Neural Network (ANN) **completely from scratch** using only Python and NumPy.  
It was inspired by my AI class, where I wanted to test how stochastic gradient descent works — without using TensorFlow or PyTorch.

## Motivation
During an AI class, my professor explained how stochastic gradient descent works.  
I wanted to test this concept manually and compare it to existing implementations.  
By randomizing each 75-sample batch from the MNIST dataset, I achieved slightly better accuracy (from **94% → 96%** after 10 epochs).

## Features
- Implemented forward & backward propagation manually  
- ReLU and Softmax activations  
- Stochastic gradient descent (SGD) optimization  
- Randomized batching for improved convergence  
- Achieves up to 96% accuracy on MNIST  

## Dataset
The model uses the [MNIST handwritten digits dataset](https://www.tensorflow.org/datasets/catalog/mnist), saved as `.npy` arrays:
- `x_train.npy`, `y_train.npy`
- `x_test.npy`, `y_test.npy`

## Run
```bash
python ann_from_scratch.py
