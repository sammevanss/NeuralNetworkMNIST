package src.mnist;

import java.io.IOException;

public class Main {

    // Configuration
    private static final int TRAIN_SIZE = 60000; // Number of training samples to load (max 60,000 in MNIST)
    private static final int TEST_SIZE = 10000; // Number of test samples to load (max 10,000 in MNIST)
    private static final int EPOCHS = 10; // Number of times to iterate through the full training set
    private static final String MODEL_FILE = "model.nn"; // Output path for saving the trained model

    // Neural network architecture: {input, hidden1, hidden2, output}
    private static final int[] LAYERS = {784, 128, 64, 10};
    // 784 = 28x28 MNIST pixels (input size)
    // 128 and 64 = hidden layers chosen for balance of performance
    // 10 = number of output classes (digits 0–9)

    /**
     * Main class for training and evaluating a simple neural network on the MNIST dataset.
     * <p>
     * This program performs the following steps:
     * 1. Loads the MNIST training and test datasets (images and labels).
     * 2. Initializes a feedforward neural network with a configurable architecture.
     * 3. Trains the network for a given number of epochs.
     * 4. Evaluates the model using test data, computing classification accuracy.
     * 5. Displays a confusion matrix showing prediction performance per digit.
     * 6. Saves the trained model to disk.
     * <p>
     * Configuration such as dataset sizes, number of epochs, model output path,
     * and network architecture can be modified at the top of this file.
     * <p>
     * Usage:
     * - Ensure MNIST binary data files are present in the `data/` directory.
     * - Run the program; training progress and results will be printed to the console.
     */
    public static void main(String[] args) throws IOException {

        // Load MNIST Data
        double[][] trainImages = MNISTLoader.loadImages("data/train-images.idx3-ubyte", TRAIN_SIZE);
        double[][] trainLabels = MNISTLoader.loadLabels("data/train-labels.idx1-ubyte", TRAIN_SIZE);
        double[][] testImages  = MNISTLoader.loadImages("data/t10k-images.idx3-ubyte", TEST_SIZE);
        double[][] testLabels  = MNISTLoader.loadLabels("data/t10k-labels.idx1-ubyte", TEST_SIZE);

        // Initialize Neural Network
        NeuralNetwork nn = new NeuralNetwork(LAYERS);

        // Train the Model
        trainModel(nn, trainImages, trainLabels, EPOCHS);

        // Evaluate Performance
        evaluateModel(nn, testImages, testLabels);

        // Save Trained Model
        nn.save(MODEL_FILE);
    }

    /**
     * Trains the neural network over a specified number of epochs.
     *
     * @param nn     The NeuralNetwork instance to train.
     * @param images The training input data (each image flattened to a 1D array).
     * @param labels The corresponding one-hot encoded labels for training data.
     * @param epochs The number of times to iterate through the entire training dataset.
     */
    private static void trainModel(NeuralNetwork nn, double[][] images, double[][] labels, int epochs) {
        // Loop over each epoch (full pass through the training set)
        for (int epoch = 0; epoch < epochs; epoch++) {
            // Iterate through each training example
            for (int i = 0; i < images.length; i++) {
                // Train the network on a single example (image and its label)
                nn.train(images[i], labels[i]);

                // Log progress every 1000 samples
                if (i % 1000 == 0) {
                    System.out.println("Trained " + i + " samples");
                }
            }
            // Indicate completion of current epoch
            System.out.println("Epoch " + (epoch + 1) + " complete.");
        }
    }

    /**
     * Evaluates the neural network's performance on test data.
     * Computes accuracy and builds a confusion matrix.
     *
     * @param nn          the trained NeuralNetwork instance
     * @param testImages  array of test images (flattened pixel data)
     * @param testLabels  corresponding one-hot encoded test labels
     */
    private static void evaluateModel(NeuralNetwork nn, double[][] testImages, double[][] testLabels) {
        int[][] confusion = new int[10][10];  // Confusion matrix: [actual][predicted]
        int correct = 0;                      // Counter for correct predictions

        // Iterate through all test samples
        for (int i = 0; i < testImages.length; i++) {
            int predicted = nn.classify(testImages[i]);   // Predict label from input image
            int actual = argMax(testLabels[i]);           // Extract actual label from one-hot encoding
            confusion[actual][predicted]++;               // Update confusion matrix cell
            if (predicted == actual) correct++;           // Increment correct count if prediction matches actual
        }

        // Print overall accuracy as a percentage
        System.out.printf("Accuracy: %.2f%%\n", 100.0 * correct / testImages.length);

        // Display confusion matrix for detailed classification results
        printConfusionMatrix(confusion);
    }

    /**
     * Finds the index of the maximum value in the given array.
     * This is commonly used to determine the predicted class from
     * a one-hot encoded output or probability distribution.
     *
     * @param array the array of values to search
     * @return the index of the highest value in the array
     */
    private static int argMax(double[] array) {
        int maxIdx = 0; // Assume the first element is the maximum initially

        // Compare it to the next element in the array
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[maxIdx]) { // Found a new maximum value
                maxIdx = i; // Update index of max value
            }
        }
        return maxIdx; // Return index of the largest element
    }


    /**
     * Prints the confusion matrix to the console in a readable format.
     * Each row represents the actual class, and each column represents the predicted class.
     * The matrix shows the count of predictions for each actual-predicted pair.
     *
     * @param matrix A 2D array where matrix[actual][predicted] holds the count of predictions.
     */
    private static void printConfusionMatrix(int[][] matrix) {
        System.out.println("Confusion Matrix:");

        for (int actualClass = 0; actualClass < matrix.length; actualClass++) {
            // Print the actual class label at the start of the row
            System.out.print(actualClass + ": ");

            for (int predictedClass = 0; predictedClass < matrix[actualClass].length; predictedClass++) {
                // Print the count of predictions for this actual-predicted pair
                System.out.printf("%4d", matrix[actualClass][predictedClass]);
            }
            System.out.println(); // Move to the next line after printing all predictions for the actual class
        }

    }
}