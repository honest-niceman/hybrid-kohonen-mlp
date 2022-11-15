package org.example.som;

import org.example.dataset.WineDataset;
import org.example.som.core.Kohonen;

import java.util.logging.Level;
import java.util.logging.Logger;

public class Main {
    private static final Logger log = Logger.getLogger("Kohonen:Main---");

    public static void main(String[] args) {
        Kohonen kohonen = new Kohonen(new WineDataset().getData());
        kohonen.trainWTA();
        double[][] testNormalize = kohonen.normalizeData(new WineDataset().getTestData());
        for (int i = 0; i < testNormalize.length; i++) {
            String result = kohonen.test(testNormalize[i]).toString();
            log.log(Level.INFO, i + ") " + result);
        }
    }
}