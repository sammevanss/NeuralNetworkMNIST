package src.mnist;

import java.io.*;

public class MNISTLoader {

    /**
     * Loads normalized MNIST images (0.0–1.0) from the specified IDX file.
     * @param filename path to the IDX image file
     * @param numImages number of images to load
     * @return a 2D array of flattened images [numImages][784]
     * @throws IOException if file cannot be read or format is invalid
     */
    public static double[][] loadImages(String filename, int numImages) throws IOException {
        try (DataInputStream in = new DataInputStream(new FileInputStream(filename))) {

            // First 4 bytes: file type identifier (2051 = image file)
            int fileType = in.readInt();
            if (fileType != 2051) {
                throw new IOException("Unexpected file type: expected 2051 for image file, got " + fileType);
            }

            int totalImages = in.readInt(); // Next 4 bytes: number of images in the file
            int rows = in.readInt();        // Next 4 bytes: number of rows per image
            int cols = in.readInt();        // Next 4 bytes: number of columns per image
            int imageSize = rows * cols;

            // Check that we’re not requesting more images than available
            if (numImages > totalImages) {
                throw new IOException("Requested more images than available (" + totalImages + ")");
            }

            // Read image data and normalize each pixel to [0.0, 1.0]
            double[][] images = new double[numImages][imageSize];
            for (int i = 0; i < numImages; i++) {
                for (int j = 0; j < imageSize; j++) {
                    images[i][j] = in.readUnsignedByte() / 255.0;
                }
            }
            return images;
        }
    }

    /**
     * Loads one-hot encoded labels from the specified IDX file.
     * @param filename path to the IDX label file
     * @param numLabels number of labels to load
     * @return a 2D array of one-hot encoded labels [numLabels][10]
     * @throws IOException if file cannot be read or format is invalid
     */
    public static double[][] loadLabels(String filename, int numLabels) throws IOException {
        try (DataInputStream in = new DataInputStream(new FileInputStream(filename))) {

            // First 4 bytes: file type identifier (2049 = label file)
            int fileType = in.readInt();
            if (fileType != 2049) {
                throw new IOException("Unexpected file type: expected 2049 for label file, got " + fileType);
            }

            int totalLabels = in.readInt(); // Next 4 bytes: number of labels in the file

            // Validate that the requested number of labels does not exceed the available count.
            // Reading fewer is allowed; exceeding would result in an out-of-bounds read.
            if (numLabels > totalLabels) {
                throw new IOException("Requested more labels than available (" + totalLabels + ")");
            }

            // Read each label and convert to one-hot encoding
            double[][] labels = new double[numLabels][10];
            for (int i = 0; i < numLabels; i++) {
                int label = in.readUnsignedByte();
                labels[i][label] = 1.0;
            }

            // Return the loaded and one-hot encoded label data
            return labels;

        }
    }
}
