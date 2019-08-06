package timeseriesweka.classifiers.distance_based.distances.wdtw;

import timeseriesweka.classifiers.distance_based.distances.DistanceMeasure;

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
    public double distance() {

        double[] a = getTarget();
        double[] b = getCandidate();
        if(seriesLength < 0 || seriesLength != a.length) {
            generateWeights(a.length);
        }
        double cutOff = getLimit();

        //create empty array
        int m = a.length;
        int n = b.length;
        double[][] distances = new double[m][n];

        //first value
        distances[0][0] = weightVector[0] * (a[0] - b[0]) * (a[0] - b[0]);

        //early abandon if first values is larger than cut off
        if (distances[0][0] > cutOff) {
            return Double.MAX_VALUE;
        }

        //top row
        for (int i = 1; i < n; i++) {
            distances[0][i] =
                distances[0][i - 1] + weightVector[i] * (a[0] - b[i]) * (a[0] - b[i]); //edited by Jay
        }

        //first column
        for (int i = 1; i < m; i++) {
            distances[i][0] =
                distances[i - 1][0] + weightVector[i] * (a[i] - b[0]) * (a[i] - b[0]); //edited by Jay
        }

        //warp rest
        double minDistance;
        for (int i = 1; i < m; i++) {
            boolean overflow = true;

            for (int j = 1; j < n; j++) {
                //calculate distances
                minDistance = Math.min(distances[i][j - 1], Math.min(distances[i - 1][j], distances[i - 1][j - 1]));
                distances[i][j] =
                    minDistance + weightVector[Math.abs(i - j)] * (a[i] - b[j]) * (a[i] - b[j]);

                if (overflow && distances[i][j] < cutOff) {
                    overflow = false; // because there's evidence that the path can continue
                }
            }

            //early abandon
            if (overflow) {
                return Double.POSITIVE_INFINITY;
            }
        }
        return distances[m - 1][n - 1];
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
