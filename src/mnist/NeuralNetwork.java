package src.mnist;

import java.io.*;

/**
 * NeuralNetwork is a basic feedforward artificial neural network implementation using
 * the sigmoid activation function and the backpropagation algorithm for supervised learning.
 * <p>
 * This implementation is suitable for classification tasks such as MNIST digit recognition.
 * <p>
 * Core features include:
 * - Multi-layer perceptron architecture
 * - Forward pass for inference
 * - Backpropagation with gradient descent for training
 * - Sigmoid activation function
 * - Model persistence via file I/O
 */
public class NeuralNetwork {
    private final int[] layerSizes;               // Number of neurons in each layer (e.g., [784, 100, 10])
    public final double[][][] weights;            // Synaptic weights[layer][neuron][prevNeuron]
    public final double[][] biases;               // Biases[layer][neuron] added before activation
    private final double[][] activations;         // Activations[layer][neuron] after applying sigmoid
    private final double[][] deltas;              // Deltas[layer][neuron] for backprop error gradients
    private final double learningRate;            // Scalar learning rate for gradient descent

    /**
     * Constructs a new neural network with specified learning rate and architecture.
     * Weights and biases are randomly initialized.
     *
     * @param learningRate step size for gradient descent (higher=faster, unstable; lower=slower, stable)
     * @param layerSizes   neuron counts per layer (input → hidden → output)
     */
    public NeuralNetwork(double learningRate, int... layerSizes) {
        this.learningRate = learningRate;
        this.layerSizes = layerSizes;
        int numLayers = layerSizes.length;

        weights = new double[numLayers][][];
        biases = new double[numLayers][];
        activations = new double[numLayers][];
        deltas = new double[numLayers][];

        for (int layer = 1; layer < numLayers; layer++) {
            int currentSize = layerSizes[layer];
            int previousSize = layerSizes[layer - 1];

            weights[layer] = new double[currentSize][previousSize];
            biases[layer] = new double[currentSize];
            activations[layer] = new double[currentSize];
            deltas[layer] = new double[currentSize];

            for (int neuron = 0; neuron < currentSize; neuron++) {
                initializeNeuron(layer, neuron, previousSize);
            }
        }
        activations[0] = new double[layerSizes[0]];
    }

    /**
     * Convenience constructor using default learning rate of 0.05.
     * @param layerSizes neuron counts per layer
     */
    public NeuralNetwork(int... layerSizes) {
        this(0.05, layerSizes);
    }

    /**
     * Helper to initialize weights and bias for a single neuron.
     */
    private void initializeNeuron(int layer, int neuron, int prevSize) {
        biases[layer][neuron] = Math.random();  // bias in [0,1)
        for (int prev = 0; prev < prevSize; prev++) {
            weights[layer][neuron][prev] = Math.random() - 0.5;  // weight in [-0.5,0.5)
        }
    }

    /**
     * Sigmoid activation function.
     */
    private double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    /**
     * Derivative of sigmoid, given sigmoid(x) output.
     */
    private double sigmoidDerivative(double x) {
        return x * (1 - x);
    }

    /**
     * Computes weighted input sum for a neuron (dot product + bias).
     */
    private double weightedSum(int layer, int neuron) {
        double sum = biases[layer][neuron];
        for (int prev = 0; prev < layerSizes[layer - 1]; prev++) {
            sum += weights[layer][neuron][prev] * activations[layer - 1][prev];
        }
        return sum;
    }

    /**
     * Activation helper: applies sigmoid to weighted sum.
     */
    private double activate(int layer, int neuron) {
        return sigmoid(weightedSum(layer, neuron));
    }

    /**
     * Performs a forward pass through the network.
     *
     * @param input input vector, must match input layer size
     * @return copy of output layer activations
     */
    public double[] predict(double[] input) {
        System.arraycopy(input, 0, activations[0], 0, input.length);
        for (int layer = 1; layer < layerSizes.length; layer++) {
            for (int neuron = 0; neuron < layerSizes[layer]; neuron++) {
                activations[layer][neuron] = activate(layer, neuron);
            }
        }
        return activations[layerSizes.length - 1].clone();
    }

    /**
     * Classifies input by returning the index of the highest activation.
     */
    public int classify(double[] input) {
        double[] output = predict(input);
        int best = 0;
        for (int i = 1; i < output.length; i++) {
            if (output[i] > output[best]) best = i;
        }
        return best;
    }

    /**
     * Trains the network on one example via backpropagation and gradient descent.
     */
    public void train(double[] input, double[] target) {
        predict(input);  // forward pass
        int out = layerSizes.length - 1;
        for (int n = 0; n < layerSizes[out]; n++) {
            double o = activations[out][n];
            deltas[out][n] = (target[n] - o) * sigmoidDerivative(o);
        }
        for (int layer = out - 1; layer > 0; layer--) {
            for (int n = 0; n < layerSizes[layer]; n++) {
                double sum = 0;
                for (int next = 0; next < layerSizes[layer + 1]; next++) {
                    sum += weights[layer + 1][next][n] * deltas[layer + 1][next];
                }
                deltas[layer][n] = sum * sigmoidDerivative(activations[layer][n]);
            }
        }
        for (int layer = 1; layer < layerSizes.length; layer++) {
            for (int n = 0; n < layerSizes[layer]; n++) {
                for (int prev = 0; prev < layerSizes[layer - 1]; prev++) {
                    weights[layer][n][prev] += learningRate * deltas[layer][n] * activations[layer - 1][prev];
                }
                biases[layer][n] += learningRate * deltas[layer][n];
            }
        }
    }

    /**
     * Serializes and saves the model to a binary file.
     */
    public void save(String filename) throws IOException {
        try (DataOutputStream out = new DataOutputStream(new FileOutputStream(filename))) {
            save(out);
        }
    }

    /**
     * Writes network parameters to a DataOutput stream.
     */
    private void save(DataOutput out) throws IOException {
        out.writeInt(layerSizes.length);
        for (int size : layerSizes) out.writeInt(size);
        for (int layer = 1; layer < layerSizes.length; layer++) {
            for (double[] neuronWeights : weights[layer]) for (double w : neuronWeights) out.writeDouble(w);
            for (double b : biases[layer]) out.writeDouble(b);
        }
    }

    /**
     * Reads network parameters from a DataInput stream.
     * @param in input stream containing model data
     * @return reconstructed NeuralNetwork
     * @throws IOException if format is invalid
     */
    private static NeuralNetwork load(DataInput in) throws IOException {
        int numLayers = in.readInt();
        int[] sizes = new int[numLayers];
        for (int i = 0; i < numLayers; i++) sizes[i] = in.readInt();
        NeuralNetwork nn = new NeuralNetwork(sizes);
        for (int layer = 1; layer < numLayers; layer++) {
            for (int n = 0; n < nn.weights[layer].length; n++) for (int p = 0; p < nn.weights[layer][n].length; p++) nn.weights[layer][n][p] = in.readDouble();
            for (int n = 0; n < nn.biases[layer].length; n++) nn.biases[layer][n] = in.readDouble();
        }
        return nn;
    }
}
