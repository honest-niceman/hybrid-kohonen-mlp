package org.example;

import org.example.dataset.IrisDataset;
import org.example.mlp.core.MultiLayerPerceptron;
import org.example.mlp.core.activationfunctions.SigmoidActivation;
import org.example.som.core.Kohonen;
import org.example.som.core.Neuron;

import java.util.*;
import java.util.logging.Level;
import java.util.logging.Logger;

public class Main {
    private static final Logger log = Logger.getLogger("Hybrid:Main---");
    private static final Random r = new Random();

    public static void main(String[] args) {
        Kohonen kohonen = new Kohonen(new IrisDataset().getData());
        double[][] kohonenData = kohonen.getData();

        kohonen.trainWTA();

        int[] layers = new int[]{3, 10, 3};
        MultiLayerPerceptron mlp = new MultiLayerPerceptron(layers, 0.1, new SigmoidActivation());

        double[][] dataAfterKohonen = new double[150][3];

        for (int i = 0; i < dataAfterKohonen.length; i++) {
            Neuron neuronWinner = kohonen.test(kohonenData[i]);
            int indexOfWinner = kohonen.getNeurons().indexOf(neuronWinner);

            if (indexOfWinner == 0) {
                Neuron loser1 = kohonen.getNeurons().get(1);
                Neuron loser2 = kohonen.getNeurons().get(2);
                dataAfterKohonen[i] = new double[]{neuronWinner.calcDistanceBetweenNeuronAndInputVector(
                        kohonenData[i]), loser1.calcDistanceBetweenNeuronAndInputVector(
                        kohonenData[i]), loser2.calcDistanceBetweenNeuronAndInputVector(kohonenData[i])};
            } else if (indexOfWinner == 1) {
                Neuron loser0 = kohonen.getNeurons().get(0);
                Neuron loser2 = kohonen.getNeurons().get(2);
                dataAfterKohonen[i] = new double[]{loser0.calcDistanceBetweenNeuronAndInputVector(
                        kohonenData[i]), neuronWinner.calcDistanceBetweenNeuronAndInputVector(
                        kohonenData[i]), loser2.calcDistanceBetweenNeuronAndInputVector(kohonenData[i])};
            } else if (indexOfWinner == 2) {
                Neuron loser0 = kohonen.getNeurons().get(0);
                Neuron loser1 = kohonen.getNeurons().get(1);
                dataAfterKohonen[i] = new double[]{loser1.calcDistanceBetweenNeuronAndInputVector(
                        kohonenData[i]), loser0.calcDistanceBetweenNeuronAndInputVector(
                        kohonenData[i]), neuronWinner.calcDistanceBetweenNeuronAndInputVector(kohonenData[i])};
            }
        }

        for (double[] doubles : dataAfterKohonen) {
            System.out.println(Arrays.toString(doubles));
        }

        double[][] normalizedKohonenData = new double[150][3];
        for (int i = 0; i < dataAfterKohonen.length; i++) {
            normalizedKohonenData[i] = normalize(dataAfterKohonen[i]);
        }

        System.out.println("norm:");
        for (double[] doubles : normalizedKohonenData) {
            System.out.println(Arrays.toString(doubles));
        }

        double[][] toShuffle = new double[150][6];
        for (int i = 0; i < normalizedKohonenData.length; i++) {
            double[] input = normalizedKohonenData[i];
            for (int j = 0; j < 6; j++) {
                double[] output;
                if (i < 50) {
                    output = new double[]{0, 0, 1};
                    toShuffle[i][0] = input[0];
                    toShuffle[i][1] = input[1];
                    toShuffle[i][2] = input[2];
                    toShuffle[i][3] = output[0];
                    toShuffle[i][4] = output[1];
                    toShuffle[i][5] = output[2];
                } else if (i > 49 && i < 100) {
                    output = new double[]{1, 0, 0};
                    toShuffle[i][0] = input[0];
                    toShuffle[i][1] = input[1];
                    toShuffle[i][2] = input[2];
                    toShuffle[i][3] = output[0];
                    toShuffle[i][4] = output[1];
                    toShuffle[i][5] = output[2];
                } else if (i > 99 && i < 150) {
                    output = new double[]{0, 1, 0};
                    toShuffle[i][0] = input[0];
                    toShuffle[i][1] = input[1];
                    toShuffle[i][2] = input[2];
                    toShuffle[i][3] = output[0];
                    toShuffle[i][4] = output[1];
                    toShuffle[i][5] = output[2];
                }
            }
        }

        double[][] shuffledDataAfterKohonen = shuffleMatrix(toShuffle);

        for (int i = 0; i < shuffledDataAfterKohonen.length; i++) {
            System.out.println(Arrays.toString(shuffledDataAfterKohonen[i]));
        }

        for (int i = 0; i < 500; i++) {
            for (int j = 0; j < 150; j++) {
                double[] input = new double[3];
                double[] output = new double[3];
                input[0] = shuffledDataAfterKohonen[j][0];
                input[1] = shuffledDataAfterKohonen[j][1];
                input[2] = shuffledDataAfterKohonen[j][2];
                output[0] = shuffledDataAfterKohonen[j][3];
                output[1] = shuffledDataAfterKohonen[j][4];
                output[2] = shuffledDataAfterKohonen[j][5];
                double error = mlp.backPropagate(input, output);
                String msg = "Iteration №" + i + ". Error = " + error;
                log.log(Level.INFO, msg);
            }
        }

        /* Test */
        for (double[] datum : shuffledDataAfterKohonen) {
            double[] toTest = new double[3];
            toTest[0] = datum[0];
            toTest[1] = datum[1];
            toTest[2] = datum[2];
            double[] ideal = new double[3];
            ideal[0] = datum[3];
            ideal[1] = datum[4];
            ideal[2] = datum[5];
            String msg = Arrays.toString(mlp.execute(toTest));
            String msg1 = Arrays.toString(ideal);
            String message = msg + "–––" + msg1;
            log.log(Level.INFO, message);
        }
    }

    public static double[][] shuffleMatrix(double [][] matrix) {
        List<double[]> rows = new ArrayList<>(Arrays.asList(matrix));
        Collections.shuffle(rows);
        for (int i = 0; i < matrix.length; i++) {
            matrix[i] = rows.get(i);
        }
        return matrix;
    }

    private static double[] normalize(double[] data) {
        int maxAt = 0;
        for (int i = 0; i < data.length; i++) {
            maxAt = data[i] > data[maxAt] ? i : maxAt;
        }
        double umax = data[maxAt];
        double[] result = new double[3];
        for (int i = 0; i < data.length; i++) {
            if (i == maxAt) {
                result[i] = 1;
                continue;
            }
            double ur = data[i];
            double diff = ur - umax;
            double diffSquare = Math.pow(diff, 2);
            double sigmaSquare = Math.pow(0.5, 2);
            double division = diffSquare / sigmaSquare;
            double makeNegative = -division;
            result[i] = Math.exp(makeNegative);
        }
        return result;
    }
}
