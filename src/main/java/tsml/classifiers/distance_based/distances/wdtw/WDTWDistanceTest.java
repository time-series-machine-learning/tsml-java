package tsml.classifiers.distance_based.distances.wdtw;

import weka.core.Instance;

public class WDTWDistanceTest {

    private static double[] generateWeights(int seriesLength, double g) {
        double halfLength = (double) seriesLength / 2;
        double[] weightVector = new double[seriesLength];
        for (int i = 0; i < seriesLength; i++) {
            weightVector[i] = 1 / (1 + Math.exp(-g * (i - halfLength)));
        }
        return weightVector;
    }

    private static double wdtwOrig(Instance a, Instance b, double limit, double g) {

        int aLength = a.numAttributes() - 1;
        int bLength = b.numAttributes() - 1;

        double[] weightVector = generateWeights(aLength, g);

        //create empty array
        double[][] distances = new double[aLength][bLength];

        //first value
        distances[0][0] = weightVector[0] * (a.value(0) - b.value(0)) * (a.value(0) - b.value(0));

        //early abandon if first values is larger than cut off
        if (distances[0][0] > limit) {
            return Double.POSITIVE_INFINITY;
        }

        //top row
        for (int i = 1; i < bLength; i++) {
            distances[0][i] =
                distances[0][i - 1] + weightVector[i] * (a.value(0) - b.value(i)) * (a.value(0) - b.value(i)); //edited by Jay
        }

        //first column
        for (int i = 1; i < aLength; i++) {
            distances[i][0] =
                distances[i - 1][0] + weightVector[i] * (a.value(i) - b.value(0)) * (a.value(i) - b.value(0)); //edited by Jay
        }

        //warp rest
        double minDistance;
        for (int i = 1; i < aLength; i++) {
            boolean overflow = true;

            for (int j = 1; j < bLength; j++) {
                //calculate distance_measures
                minDistance = Math.min(distances[i][j - 1], Math.min(distances[i - 1][j], distances[i - 1][j - 1]));
                distances[i][j] =
                    minDistance + weightVector[Math.abs(i - j)] * (a.value(i) - b.value(j)) * (a.value(i) - b.value(j));

                if (overflow && distances[i][j] < limit) {
                    overflow = false; // because there's evidence that the path can continue
                }
            }

            //early abandon
            if (overflow) {
                return Double.POSITIVE_INFINITY;
            }
        }
        return distances[aLength - 1][bLength - 1];
    }
}
