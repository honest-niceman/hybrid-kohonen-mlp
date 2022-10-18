package org.example.som;

import org.example.dataset.IrisDataset;
import org.example.som.core.Kohonen;

import java.util.logging.Level;
import java.util.logging.Logger;

public class Main {
    private static final Logger log = Logger.getLogger("Kohonen:Main---");

    public static void main(String[] args) {
        Kohonen kohonen = new Kohonen(new IrisDataset().getData());
        kohonen.trainWTA();
        double[][] normalize = kohonen.normalizeData(new IrisDataset().getData());
        for (double[] doubles : normalize) {
            String result = kohonen.test(doubles).toString();
            log.log(Level.INFO, result);
        }
    }
}