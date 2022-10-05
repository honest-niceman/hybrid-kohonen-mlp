package org.example.som.core;

import java.util.Arrays;
import java.util.Random;

public class Neuron {
    public static final double MIN = 0.0;
    public static final double MAX = 0.5;
    private final Random r = new Random();
    double[] arrayOfWeight;

    public Neuron(int countAttrs) {
        arrayOfWeight = new double[countAttrs];
        for (int a = 0; a < countAttrs; a++) {
            arrayOfWeight[a] = MIN + (MAX - MIN) * r.nextDouble();
        }
    }

    // эвклидова мера
    public double calcDistanceBetweenNeuronAndInputVector(double[] inputVector) {
        double currentDistanceToNeuron = 0;
        for (int i = 0; i < arrayOfWeight.length; i++) {
            currentDistanceToNeuron += Math.pow(inputVector[i] - arrayOfWeight[i], 2);
        }
        return Math.sqrt(currentDistanceToNeuron);
    }

    @Override
    public String toString() {
        return "Neuron{" +
                "arrayOfWeight=" + Arrays.toString(arrayOfWeight) +
                '}';
    }
}