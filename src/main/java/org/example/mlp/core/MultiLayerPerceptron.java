package org.example.mlp.core;

import org.example.mlp.core.activationfunctions.interfaces.ActivationFunction;

public class MultiLayerPerceptron {
    private final double learningRate;
    private final ActivationFunction activationFunction;
    private final Layer[] layers;

    public MultiLayerPerceptron(int[] layers, double learningRate, ActivationFunction fun) {
        this.learningRate = learningRate;
        this.activationFunction = fun;
        this.layers = new Layer[layers.length];

        this.layers[0] = new Layer(layers[0], 0);
        for (int i = 1; i < layers.length; i++) {
            this.layers[i] = new Layer(layers[i], layers[i - 1]);
        }
    }

    public double[] execute(double[] input) {
        double newValue;
        double[] output = new double[layers[layers.length - 1].getLength()];
        // Put input
        for (int i = 0; i < layers[0].getLength(); i++) {
            layers[0].getNeurons()[i].setValue(input[i]);
        }
        // Execute - hiddens + output
        for (int k = 1; k < layers.length; k++) {
            for (int i = 0; i < layers[k].getLength(); i++) {
                newValue = 0.0;
                for (int j = 0; j < layers[k - 1].getLength(); j++) {
                    newValue += layers[k].getNeurons()[i].getWeights()[j] * layers[k - 1].getNeurons()[j].getValue();
                }
                newValue += layers[k].getNeurons()[i].getBias();
                layers[k].getNeurons()[i].setValue(activationFunction.activate(newValue));
            }
        }
        // Get output
        for (int i = 0; i < layers[layers.length - 1].getLength(); i++) {
            output[i] = layers[layers.length - 1].getNeurons()[i].getValue();
        }
        return output;
    }

    /**
     * Backpropagation algorithm. Assisted for learning method.
     * <p>
     * Non-guaranteed and very slow convergence; use as stop criteria
     * a norm between previous and current errors, and a maximum number
     * of iterations.
     *
     * @param input  Input values (scaled between 0 and 1)
     * @param output Expected output values (scaled between 0 and 1)
     * @return Delta error between output generated and expected output
     */
    public double backPropagate(double[] input, double[] output) {
        double[] newOutput = execute(input);
        double error;
        // Calculate the output error
        for (int i = 0; i < layers[layers.length - 1].getLength(); i++) {
            error = output[i] - newOutput[i];
            layers[layers.length - 1].getNeurons()[i].setDelta(error * activationFunction.derivative(newOutput[i]));
        }
        for (int k = layers.length - 2; k >= 0; k--) {
            // Calculate the error of the current layer and recalculate the Deltas
            for (int i = 0; i < layers[k].getLength(); i++) {
                error = 0.0;
                for (int j = 0; j < layers[k + 1].getLength(); j++) {
                    error += layers[k + 1].getNeurons()[j].getDelta() * layers[k + 1].getNeurons()[j].getWeights()[i];
                }
                layers[k].getNeurons()[i].setDelta(error * activationFunction.derivative(layers[k].getNeurons()[i].getValue()));
            }
            // Update the weights of the next layer
            for (int i = 0; i < layers[k + 1].getLength(); i++) {
                for (int j = 0; j < layers[k].getLength(); j++) {
                    double[] weights = layers[k + 1].getNeurons()[i].getWeights();
                    weights[j] += learningRate * layers[k + 1].getNeurons()[i].getDelta() * layers[k].getNeurons()[j].getValue();
                    layers[k + 1].getNeurons()[i].setWeights(weights);
                }
                double bias = layers[k + 1].getNeurons()[i].getBias();
                bias = bias + learningRate * layers[k + 1].getNeurons()[i].getDelta();
                layers[k + 1].getNeurons()[i].setBias(bias);
            }
        }
        error = 0.0;
        for (int i = 0; i < output.length; i++) {
            error += Math.abs(newOutput[i] - output[i]);
        }
        return error / output.length;
    }
}