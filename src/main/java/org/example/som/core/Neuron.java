package org.example.som.core;

import java.util.Arrays;
import java.util.Random;

public class Neuron {
    private final Random r = new Random();
    private final double[] weights;

    public Neuron(int countAttrs) {
        weights = new double[countAttrs];
        for (int a = 0; a < countAttrs; a++) {
            weights[a] = r.nextDouble();
        }
    }

    // Евклидово расстояние https://en.wikipedia.org/wiki/Euclidean_distance
    public double calcDistanceBetweenNeuronAndInputVector(double inputVector) {
        double currentDistanceToNeuron = 0;
        for (int i = 0; i < weights.length; i++) {
            currentDistanceToNeuron += Math.pow(inputVector - weights[i], 2);
        }
        return Math.sqrt(currentDistanceToNeuron);
    }

    public double[] getWeights() {
        return weights;
    }

    public void setWeight(int idx, double weight) {
        this.weights[idx] = weight;
    }

    @Override
    public String toString() {
        return String.valueOf(weights[0]);
    }
}