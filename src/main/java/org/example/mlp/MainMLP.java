package org.example.mlp;

import org.example.dataset.ForecastDataset;
import org.example.mlp.core.MultiLayerPerceptron;
import org.example.mlp.core.activationfunctions.SigmoidActivation;

import java.util.logging.Level;
import java.util.logging.Logger;

public class MainMLP {
    private static final Logger log = Logger.getLogger("MLP:Main---");
    private double[] trainedData;

    private final int[] layers = new int[]{1, 3, 1};
    private final MultiLayerPerceptron mlp = new MultiLayerPerceptron(layers, 0.5, new SigmoidActivation());

    public void train() {
        /* Learn */
        double[] idealData = new ForecastDataset().getSinData();
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < idealData.length - 1; j++) {
                double error = mlp.backPropagate(new double[] {trainedData[i]}, new double[] {idealData[i] });
                String msg = "Iteration №" + i + ". Error = " + error;
                log.log(Level.INFO, msg);
            }
        }
        log.log(Level.INFO, "Learning completed!");
    }

    public void setTrainedData(double[] trainedData) {
        this.trainedData = trainedData;
    }

    public MultiLayerPerceptron getMlp() {
        return mlp;
    }
}