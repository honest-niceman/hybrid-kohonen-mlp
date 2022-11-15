package org.example.som.core;

import java.util.ArrayList;
import java.util.List;

public class Kohonen {
    private static final int NEURONS_NUMBER = 3;
    private static final int COUNT_TRAIN_ITERATIONS = 1000;
    private static final double N_DEFAULT = 0.5;
    private static final int COUNT_ATTRIBUTES_IN_VECTOR = 13;

    private final List<Neuron> neurons;
    private final double[][] data;

    public Kohonen(double[][] data) {
        neurons = new ArrayList<>(NEURONS_NUMBER);
        for (int i = 0; i < NEURONS_NUMBER; i++) {
            neurons.add(new Neuron(COUNT_ATTRIBUTES_IN_VECTOR));
        }
        this.data = normalizeData(data);
    }

    // обучение по алгоритму победитель получает всё
    public void trainWTA() {
        double n = N_DEFAULT;
        for (int t = 0; t < COUNT_TRAIN_ITERATIONS; t++) {
            for (int i = 0; i < data.length; i++) {
                double[] datum = data[i];
                if (i < 24 + 12) {
                    trainNeuronWTA(neurons.get(0), datum, n);
                } else if (i > 47 - 1 && i < 70 + 12) {
                    trainNeuronWTA(neurons.get(1), datum, n);
                } else if (i > 107 - 1 && i < 130 + 12) {
                    trainNeuronWTA(neurons.get(2), datum, n);
                }
            }
        }
    }

    private Neuron findNeuronWinnerWTA(double[] vector) {
        Neuron neuronWithMinDistance = neurons.get(0);
        for (int i = 0; i < NEURONS_NUMBER; i++) {
            double winnerDistance = neuronWithMinDistance.calcDistanceBetweenNeuronAndInputVector(vector);
            double iDistance = neurons.get(i).calcDistanceBetweenNeuronAndInputVector(vector);
            if (winnerDistance > iDistance) {
                neuronWithMinDistance = neurons.get(i);
            }
        }
        return neuronWithMinDistance;
    }


    public void trainNeuronWTA(Neuron neuronWinner, double[] inputVector, double n) {
        for (int w = 0; w < neuronWinner.getWeights().length; w++) {
            double newWeight = neuronWinner.getWeights()[w] + n * (inputVector[w] - neuronWinner.getWeights()[w]);
            neuronWinner.setWeight(w, newWeight);
        }
    }

    public double[][] normalizeData(double[][] data) {
        double[][] result = new double[data.length][data[0].length];
        for (int i = 0; i < data.length; i++) {
            double divider = calcDividerForNormalization(data[i]);
            for (int j = 0; j < COUNT_ATTRIBUTES_IN_VECTOR; j++) {
                result[i][j] = data[i][j] / divider;
            }
        }
        return result;
    }

    private double calcDividerForNormalization(double[] vector) {
        double result = 0;
        for (int i = 0; i < COUNT_ATTRIBUTES_IN_VECTOR; i++) {
            result += Math.pow(vector[i], 2);
        }
        return Math.sqrt(result);
    }

    public Neuron test(double[] vector) {
        return findNeuronWinnerWTA(vector);
    }

    public List<Neuron> getNeurons() {
        return neurons;
    }

    public double[][] getData() {
        return data;
    }
}
