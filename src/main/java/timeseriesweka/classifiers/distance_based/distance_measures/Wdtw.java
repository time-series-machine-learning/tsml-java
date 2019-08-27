package timeseriesweka.classifiers.distance_based.distance_measures;

import weka.core.Instance;

public class Wdtw
    extends DistanceMeasure {

    public double getG() {
        return g;
    }

    public void setG(double g) {
        this.g = g;
    }

    private double g = 0.05;

    private void generateWeights(int length) {
        seriesLength = length;
        double halfLength = (double) seriesLength / 2;
        weightVector = new double[seriesLength];
        for (int i = 0; i < seriesLength; i++) {
            weightVector[i] = 1 / (1 + Math.exp(-g * (i - halfLength)));
        }
    }

    private int seriesLength = -1;
    private double[] weightVector;

    @Override
    public double measureDistance() {
        
        Instance a = getFirstInstance();
        int aLength = a.numAttributes() - 1;
        Instance b = getSecondInstance();
        int bLength = b.numAttributes() - 1;
        if(seriesLength < 0 || seriesLength != aLength) {
            generateWeights(aLength);
        }
        double cutOff = getLimit();

        //create empty array
        double[][] distances = new double[aLength][bLength];

        //first value
        distances[0][0] = weightVector[0] * (a.value(0) - b.value(0)) * (a.value(0) - b.value(0));

        //early abandon if first values is larger than cut off
        if (distances[0][0] > cutOff) {
            return Double.MAX_VALUE;
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

                if (overflow && distances[i][j] < cutOff) {
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


    public static final String WEIGHT_KEY = "weight";

    @Override
    public void setOption(final String key, final String value) {
        if (key.equals(WEIGHT_KEY)) {
            setG(Double.parseDouble(value));
        }
    }

    @Override
    public String[] getOptions() {
        return new String[] {
            WEIGHT_KEY,
            String.valueOf(g),
            };
    }


    public static final String NAME = "WDTW";

    @Override
    public String toString() {
        return NAME;
    }

}
