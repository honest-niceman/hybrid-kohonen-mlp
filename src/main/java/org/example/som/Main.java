package org.example.som;

import org.example.som.core.NeuralNetwork;

import java.util.logging.Level;
import java.util.logging.Logger;

public class Main {
    private static final Logger log = Logger.getLogger("Main");

    public static void main(String[] args) {
        NeuralNetwork neuralNetwork = new NeuralNetwork(Dataset.data);
        neuralNetwork.trainWTA();
        double[][] normalize = neuralNetwork.normalizeData(Dataset.data);
        for (double[] doubles : normalize) {
            String result = neuralNetwork.test(doubles).toString();
            log.log(Level.INFO, result);
        }
    }
}