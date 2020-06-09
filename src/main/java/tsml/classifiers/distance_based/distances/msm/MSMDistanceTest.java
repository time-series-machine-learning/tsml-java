package tsml.classifiers.distance_based.distances.msm;

import weka.core.Instance;

public class MSMDistanceTest {

    private static double findCost(double newPoint, double x, double y, double c) {
        double dist = 0;

        if(((x <= newPoint) && (newPoint <= y)) ||
            ((y <= newPoint) && (newPoint <= x))) {
            dist = c;
        } else {
            dist = c + Math.min(Math.abs(newPoint - x), Math.abs(newPoint - y));
        }

        return dist;
    }

    private static double msmOrig(Instance a, Instance b, double limit, double c) {

        int aLength = a.numAttributes() - 1;
        int bLength = b.numAttributes() - 1;

        double[][] cost = new double[aLength][bLength];

        // Initialization
        cost[0][0] = Math.abs(a.value(0) - b.value(0));
        for(int i = 1; i < aLength; i++) {
            cost[i][0] = cost[i - 1][0] + findCost(a.value(i), a.value(i - 1), b.value(0), c);
        }
        for(int i = 1; i < bLength; i++) {
            cost[0][i] = cost[0][i - 1] + findCost(b.value(i), a.value(0), b.value(i - 1), c);
        }

        // Main Loop
        double min;
        for(int i = 1; i < aLength; i++) {
            min = limit;
            for(int j = 1; j < bLength; j++) {
                double d1, d2, d3;
                d1 = cost[i - 1][j - 1] + Math.abs(a.value(i) - b.value(j));
                d2 = cost[i - 1][j] + findCost(a.value(i), a.value(i - 1), b.value(j), c);
                d3 = cost[i][j - 1] + findCost(b.value(j), a.value(i), b.value(j - 1), c);
                cost[i][j] = Math.min(d1, Math.min(d2, d3));

                if(cost[i][j] >= limit) {
                    cost[i][j] = Double.POSITIVE_INFINITY;
                }

                if(cost[i][j] < min) {
                    min = cost[i][j];
                }
            }
            if(min >= limit) {
                return Double.POSITIVE_INFINITY;
            }
        }
        // Output
        return cost[aLength - 1][bLength - 1];
    }
}
