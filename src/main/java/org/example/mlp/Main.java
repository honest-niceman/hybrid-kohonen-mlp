package org.example.mlp;

import org.example.dataset.IrisDataset;
import org.example.mlp.core.MultiLayerPerceptron;
import org.example.mlp.core.activationfunctions.impl.SigmoidActivation;

import java.util.Arrays;
import java.util.logging.Level;
import java.util.logging.Logger;

public class Main {
    private static final Logger log = Logger.getLogger("MLP:Main---");

    public static void main(String[] args) {
        int[] layers = new int[]{4, 10, 10, 3};
        MultiLayerPerceptron mlp = new MultiLayerPerceptron(layers, 0.6, new SigmoidActivation());
        /* Learn */
        double[][] data = new IrisDataset().getData();
        for (int i = 0; i < 100; i++) {
            for (int j = 0; j < data.length; j++) {
                double[] input;
                double[] output;
                if (j < 50) {
                    input = data[j];
                    output = new double[]{1, 0, 0};
                    double error = mlp.backPropagate(input, output);
                    String msg = "Iteration №" + i + ". Error = " + error;
                    log.log(Level.INFO, msg);
                } else if (j > 49 && j < 100) {
                    input = data[j];
                    output = new double[]{0, 1, 0};
                    double error = mlp.backPropagate(input, output);
                    String msg = "Iteration №" + i + ". Error = " + error;
                    log.log(Level.INFO, msg);
                } else if (j > 99 && j < 150) {
                    input = data[j];
                    output = new double[]{0, 0, 1};
                    double error = mlp.backPropagate(input, output);
                    String msg = "Iteration №" + i + ". Error = " + error;
                    log.log(Level.INFO, msg);
                }
            }
        }
        log.log(Level.INFO, "Learning completed!");
        /* Test */
        for (double[] datum : data) {
            String msg = Arrays.toString(mlp.execute(datum));
            log.log(Level.INFO, msg);
        }
    }
}