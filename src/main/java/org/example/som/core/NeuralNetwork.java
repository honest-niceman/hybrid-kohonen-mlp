package org.example.som.core;

public class NeuralNetwork {

    private static final int NEURONS_NUMBER = 3;
    private static final int COUNT_TRAIN_ITERATIONS = 1000;
    private static final double N_DEFAULT = 0.5;
    private static final int COUNT_ATTRIBUTES_IN_VECTOR = 4;

    private final Neuron[] neurons;
    private final double[][] data;

    public NeuralNetwork(double[][] data) {
        neurons = new Neuron[NEURONS_NUMBER];
        for (int i = 0; i < NEURONS_NUMBER; i++) {
            neurons[i] = new Neuron(COUNT_ATTRIBUTES_IN_VECTOR);
        }
        this.data = normalizeData(data);
    }

    // обучение по алгоритму победитель получает всё
    public void trainWTA() {
        double n = N_DEFAULT;
        for (int t = 0; t < COUNT_TRAIN_ITERATIONS; t++) {
            for (int i = 0; i < data.length; i++) {
                double[] datum = data[i];
                if (i < 50) {
                    trainNeuronWTA(neurons[0], datum, n);
                } else if (i > 49 && i < 100) {
                    trainNeuronWTA(neurons[1], datum, n);
                } else if (i > 99 && i < 150) {
                    trainNeuronWTA(neurons[2], datum, n);
                }
            }
        }
    }

    private Neuron findNeuronWinnerWTA(double[] vector) {
        Neuron neuronWithMinDistance = neurons[0];
        for (int i = 0; i < NEURONS_NUMBER; i++) {
            double winnerDistance = neuronWithMinDistance.calcDistanceBetweenNeuronAndInputVector(vector);
            double iDistance = neurons[i].calcDistanceBetweenNeuronAndInputVector(vector);
            if (winnerDistance > iDistance) {
                neuronWithMinDistance = neurons[i];
            }
        }
        return neuronWithMinDistance;
    }


    public void trainNeuronWTA(Neuron neuronWinner, double[] inputVector, double n) {
        for (int w = 0; w < neuronWinner.getWeights().length; w++) {
            neuronWinner.getWeights()[w] = neuronWinner.getWeights()[w] + n * (inputVector[w] - neuronWinner.getWeights()[w]);
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
}
