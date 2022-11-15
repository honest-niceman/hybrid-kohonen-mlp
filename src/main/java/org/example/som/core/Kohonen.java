package org.example.som.core;

import java.util.ArrayList;
import java.util.List;

public class Kohonen {
    private static final int NEURONS_NUMBER = 5;
    private static final int COUNT_TRAIN_ITERATIONS = 100;
    private static final double N_DEFAULT = 0.5;
    private static final int COUNT_ATTRIBUTES_IN_VECTOR = 1;

    private final List<Neuron> neurons;
    private final double[] data;

    public Kohonen(double[] data) {
        neurons = new ArrayList<>(NEURONS_NUMBER);
        for (int i = 0; i < NEURONS_NUMBER; i++) {
            neurons.add(new Neuron(COUNT_ATTRIBUTES_IN_VECTOR));
        }
        this.data = data;
    }

    // обучение по алгоритму победитель получает всё
    public void trainWTA() {
        for (int t = 0; t < COUNT_TRAIN_ITERATIONS; t++) {
            int currentStart = 0;
            for (int i = 0; i < data.length; i++) {
                if (i + currentStart > 299) break;
                for (int j = 0; j < COUNT_ATTRIBUTES_IN_VECTOR; j++) {
                    double datum = data[currentStart + j];
                    trainNeuronWTA(neurons.get(j), datum, N_DEFAULT);
                }
                currentStart++;
            }
        }
    }

    private Neuron findNeuronWinnerWTA(double vector) {
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


    public void trainNeuronWTA(Neuron neuronWinner, double inputVector, double n) {
        for (int w = 0; w < neuronWinner.getWeights().length; w++) {
            double newWeight = neuronWinner.getWeights()[w] + n * (inputVector - neuronWinner.getWeights()[w]);
            neuronWinner.setWeight(w, newWeight);
        }
    }

    public Neuron test(double vector) {
        return findNeuronWinnerWTA(vector);
    }
}
