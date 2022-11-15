package org.example.som;

import org.example.dataset.ForecastDataset;
import org.example.som.core.Kohonen;
import org.example.som.core.Neuron;

public class MainKohonen {
    public static void main(String[] args) {
        double[] data1 = new ForecastDataset().getSinData();
        Kohonen kohonen1 = new Kohonen(data1);
        kohonen1.trainWTA();

        double[] trainedOutput = new double[300];
        for (int i = 0; i < 300; i++) {
            Neuron result1 = kohonen1.test(data1[i]);
            System.out.println("Sin: " + result1 + "; real value:" + data1[i]);
            trainedOutput[i] = result1.getWeights()[0];
        }
    }
}