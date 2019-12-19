package tsml.classifiers.distance_based.distances;

import utilities.StrUtils;
import weka.core.Instance;
import weka.core.neighboursearch.PerformanceStats;

import java.util.ArrayList;
import java.util.Collections;

public class Wdtw
    extends AbstractDistanceMeasure {

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
    public double distance(final Instance first,
                           final Instance second,
                           final double limit,
                           final PerformanceStats stats) {

        checks(first, second);

        int aLength = first.numAttributes() - 1;
        int bLength = second.numAttributes() - 1;
        if(seriesLength < 0 || seriesLength != aLength) {
            generateWeights(aLength);
        }

        //create empty array
        double[][] distances = new double[aLength][bLength];

        //first value
        distances[0][0] = weightVector[0] * (first.value(0) - second.value(0)) * (first.value(0) - second.value(0));

        //early abandon if first values is larger than cut off
        if (distances[0][0] > limit) {
            return Double.POSITIVE_INFINITY;
        }

        //top row
        for (int i = 1; i < bLength; i++) {
            distances[0][i] =
                distances[0][i - 1] + weightVector[i] * (first.value(0) - second.value(i)) * (first.value(0) - second.value(i)); //edited by Jay
        }

        //first column
        for (int i = 1; i < aLength; i++) {
            distances[i][0] =
                distances[i - 1][0] + weightVector[i] * (first.value(i) - second.value(0)) * (first.value(i) - second.value(0)); //edited by Jay
        }

        //warp rest
        double minDistance;
        for (int i = 1; i < aLength; i++) {
            boolean overflow = true;

            for (int j = 1; j < bLength; j++) {
                //calculate distance_measures
                minDistance = Math.min(distances[i][j - 1], Math.min(distances[i - 1][j], distances[i - 1][j - 1]));
                distances[i][j] =
                    minDistance + weightVector[Math.abs(i - j)] * (first.value(i) - second.value(j)) * (first.value(i) - second.value(j));

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


    public static final String G_FLAG = "g";

    public void setOption(final String key, final String value) {
        if (key.equals(G_FLAG)) {
            setG(Double.parseDouble(value));
        }
    }

    @Override
    public void setOptions(final String[] options) throws
                                                   Exception {
        super.setOptions(options);
        StrUtils.setOption(options, G_FLAG, this::setG, Double::parseDouble);
    }

    @Override
    public String[] getOptions() {
        ArrayList<String> options = new ArrayList<>();
        StrUtils.addOption(G_FLAG, options, g);
        Collections.addAll(options, super.getOptions());
        return options.toArray(new String[0]);
    }

}
