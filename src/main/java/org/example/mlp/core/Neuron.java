package org.example.mlp.core;

import java.util.Random;

public class Neuron {
    private double value;
    private double[] weights;
    private double bias;
    private double delta;

    Random r = new Random();
    private static final double RANGE_MIN = -0.3;
    private static final double RANGE_MAX = 0.3;

    public Neuron(int prevLayerSize) {
        bias = 0.5;
        delta = Math.random() * 0.3 + 0.1;
        value = Math.random() * 0.3 + 0.1;

        weights = new double[prevLayerSize];
        for (int i = 0; i < weights.length; i++) {
            weights[i] = RANGE_MIN + (RANGE_MAX - RANGE_MIN) * r.nextDouble();
        }
    }

    public double getValue() {
        return value;
    }

    public void setValue(double value) {
        this.value = value;
    }

    public double[] getWeights() {
        return weights;
    }

    public void setWeights(double[] weights) {
        this.weights = weights;
    }

    public double getBias() {
        return bias;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }

    public double getDelta() {
        return delta;
    }

    public void setDelta(double delta) {
        this.delta = delta;
    }
}