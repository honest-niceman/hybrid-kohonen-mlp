package org.example.som;

import org.example.som.core.NeuralNetwork;

public class Main {
    public static void main(String[] args) {
        NeuralNetwork neuralNetwork = new NeuralNetwork(Dataset.data);
        neuralNetwork.trainWTA();
        double[][] normalize = neuralNetwork.normalizeData(Dataset.data);
        for (double[] doubles : normalize) {
            System.out.println(neuralNetwork.test(doubles).toString());
        }
    }
}