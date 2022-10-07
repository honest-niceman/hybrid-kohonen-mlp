package org.example.mlp.core;

public class Neuron {
    private double value;
    private double[] weights;
    private double bias;
    private double delta;

    public Neuron(int prevLayerSize) {
        bias = Math.random() * 0.3 + 0.1;
        delta = Math.random() * 0.3 + 0.1;
        value = Math.random() * 0.3 + 0.1;

        weights = new double[prevLayerSize];
        for (int i = 0; i < weights.length; i++) {
            weights[i] = Math.random() * 0.3 + 0.1;
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