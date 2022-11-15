package org.example;

import org.example.dataset.ForecastDataset;
import org.example.dataset.RealForecastDataset;
import org.example.mlp.MainMLP;
import org.example.mlp.core.MultiLayerPerceptron;
import org.example.som.core.Kohonen;
import org.example.som.core.Neuron;

import java.util.Arrays;

public class Main {
    private static final double d2 = 1;
    private static final double d1 = 0;
    private static double maxEl;
    private static double minEl;

//    public static void main(String[] args) {
//
//        double[] data1 = new ForecastDataset().getSinData();
//        Kohonen kohonen1 = new Kohonen(data1);
//        kohonen1.trainWTA();
//
//        double[] trainedOutput = new double[300];
//        for (int i = 0; i < 300; i++) {
//            Neuron result1 = kohonen1.test(data1[i]);
//            System.out.println("Sin: " + result1 + "; real value:" + data1[i]);
//            trainedOutput[i] = result1.getWeights()[0];
//        }
//
//        MainMLP mlp = new MainMLP();
//        mlp.setTrainedData(trainedOutput);
//        mlp.setIdealData(new ForecastDataset().getSinData());
//        mlp.train();
//
//        MultiLayerPerceptron multiLayerPerceptron = mlp.getMlp();
//
//        double[] result = multiLayerPerceptron.execute(new double[]{data1[298]});
//        System.out.println("Predicted: " + result[0] + "; Real:" + data1[298]);
//    }

    public static void main(String[] args) {
        double[] data1 = new RealForecastDataset().getAppleStocks();

        double[] normalizedData = normalizeData(data1);

        Kohonen kohonen1 = new Kohonen(normalizedData);
        kohonen1.trainWTA();

        double[] trainedOutput = new double[250];
        for (int i = 0; i < 250; i++) {
            Neuron result1 = kohonen1.test(normalizedData[i]);
            System.out.println("Sin: " + result1 + "; real value:" + normalizedData[i]);
            trainedOutput[i] = result1.getWeights()[0];
        }

        MainMLP mlp = new MainMLP();
        mlp.setTrainedData(trainedOutput);
        mlp.setIdealData(normalizedData);
        mlp.train();

        MultiLayerPerceptron multiLayerPerceptron = mlp.getMlp();

        double[] predictedResult = new double[250];
        for (int i = 0; i < predictedResult.length; i++) {
            double[] result = multiLayerPerceptron.execute(new double[]{normalizedData[i]});
            predictedResult[i] = result[0];
        }

        System.out.println(deNormalizeData(predictedResult)[249]);
    }

    public static double[] normalizeData(double[] data) {
        maxEl = Arrays.stream(data).max().orElse(0);
        minEl = Arrays.stream(data).min().orElse(0);
        double[] normData = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            normData[i] = ((data[i] - minEl) / (maxEl - minEl)) + d1;
        }
        return normData;
    }

    public static double[] deNormalizeData(double[] normData) {
        double[] data = new double[normData.length];
        for (int i = 0; i < data.length; i++) {
            data[i] = ((normData[i] - d1) * (maxEl - minEl)) + minEl;
        }
        return data;
    }
}
