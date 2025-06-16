# Neural Network MNIST Classifier

This project is a simple implementation of a **feedforward neural network** in pure Java (no external libraries), trained on the MNIST dataset.

It supports:

* Training from raw MNIST binary files
* Evaluation on test data
* Displaying a confusion matrix
* Saving and loading trained models

---

## Network Architecture

The neural network uses a fully connected feedforward architecture with configurable layers.

Hard Coded Values:
```
Input Layer:     784 nodes (28x28 pixels)  
Hidden Layer 1:  128 nodes  
Hidden Layer 2:   64 nodes  
Output Layer:     10 nodes (digits 0–9)
```

---

## Project Structure

```
src/
└── mnist/
    ├── Main.java                 # Entry point: trains, evaluates, and saves the model  
    ├── MNISTLoader.java          # Loads MNIST image and label data  
    └── NeuralNetwork.java        # Core implementation of the neural network

data/                             # MNIST binary data files (not included)
├── train-images-idx3-ubyte 
├── train-labels-idx1-ubyte
├── t10k-images-idx3-ubyte
└── t10k-labels-idx1-ubyte

model.nn                          # Saved model file generated after training
```
---

## Requirements
* Java 8 or higher
* MNIST binary data files

---

## Program Flow
1. **Load and preprocess** the MNIST data
2. **Train** the neural network over multiple epochs
3. **Evaluate** the model on the test dataset
4. **Display** accuracy and a confusion matrix
5. **Save** the model weights to `model.nn`

---

## Sample Output

```

Trained 0 samples
Trained 1000 samples
Trained 2000 samples
...
Trained 59000 samples
Epoch 1 complete.
Trained 0 samples
Trained 1000 samples
Trained 2000 samples
...
Trained 58000 samples
Trained 59000 samples
Epoch 5 complete.
Trained 0 samples
Trained 1000 samples
...
Trained 56000 samples
Trained 57000 samples
Trained 58000 samples
Trained 59000 samples
Epoch 10 complete.
Accuracy: 96.97%
Confusion Matrix:
0:  971   0   1   0   0   2   3   1   2   0
1:    01125   2   1   0   1   3   2   1   0
2:    7   2 994   7   5   1   4   7   5   0
3:    2   1   5 983   0   5   1   7   4   2
4:    1   0   2   0 956   1   6   2   2  12
5:    4   2   0  14   1 851   9   1   7   3
6:    8   3   0   0   4   7 934   1   1   0
7:    1  13  11   3   3   1   0 989   1   6
8:    6   1   2   9   5   1   6   5 938   1
9:    8   4   0  12  10   4   0  10   5 956
```