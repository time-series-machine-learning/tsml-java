package tsml.classifiers.distance_based.distances;

import utilities.params.ParamHandler;
import utilities.params.ParamSet;
import weka.core.Instance;
import weka.core.neighboursearch.PerformanceStats;

public class ERPDistance extends AbstractDistanceMeasure {

    private double penalty = 0;

    public double getPenalty() {
        return penalty;
    }

    public void setPenalty(double g) {
        this.penalty = g;
    }

    @Override
    public double distance(final Instance first,
                           final Instance second,
                           final double limit,
                           final PerformanceStats stats) {

        checkData(first, second);

        int aLength = first.numAttributes() - 1;
        int bLength = second.numAttributes() - 1;

        // Current and previous columns of the matrix
        double[] curr = new double[bLength];
        double[] prev = new double[bLength];

        // size of edit distance band
        // bandsize is the maximum allowed distance to the diagonal
//        int band = (int) Math.ceil(v2.getDimensionality() * bandSize);
        int band = getBandSize();
        if (band < 0) {
            band = aLength + 1;
        }

        // g parameters for local usage
        double gValue = penalty;

        for (int i = 0;
             i < aLength;
             i++) {
            // Swap current and prev arrays. We'll just overwrite the new curr.
            {
                double[] temp = prev;
                prev = curr;
                curr = temp;
            }
            int l = i - (band + 1);
            if (l < 0) {
                l = 0;
            }
            int r = i + (band + 1);
            if (r > (bLength - 1)) {
                r = (bLength - 1);
            }

            boolean tooBig = true;

            for (int j = l;
                 j <= r;
                 j++) {
                if (Math.abs(i - j) <= band) {
                    // compute squared distance of feature vectors
                    double val1 = first.value(i);
                    double val2 = gValue;
                    double diff = (val1 - val2);
                    final double dist1 = diff * diff;

                    val1 = gValue;
                    val2 = second.value(j);
                    diff = (val1 - val2);
                    final double dist2 = diff * diff;

                    val1 = first.value(i);
                    val2 = second.value(j);
                    diff = (val1 - val2);
                    final double dist12 = diff * diff;

                    final double cost;

                    if ((i + j) != 0) {
                        if ((i == 0) || ((j != 0) && (((prev[j - 1] + dist12) > (curr[j - 1] + dist2)) && ((curr[j - 1] + dist2) < (prev[j] + dist1))))) {
                            // del
                            cost = curr[j - 1] + dist2;
                        } else if ((j == 0) || ((i != 0) && (((prev[j - 1] + dist12) > (prev[j] + dist1)) && ((prev[j] + dist1) < (curr[j - 1] + dist2))))) {
                            // ins
                            cost = prev[j] + dist1;
                        } else {
                            // match
                            cost = prev[j - 1] + dist12;
                        }
                    } else {
                        cost = 0;
                    }

                    curr[j] = cost;

                    if (tooBig && cost < limit) {
                        tooBig = false;
                    }
                } else {
                    curr[j] = Double.POSITIVE_INFINITY; // outside band
                }
            }
            if (tooBig) {
                return Double.POSITIVE_INFINITY;
            }
        }

        return curr[bLength - 1];
    }


    public static final String PENALTY_FLAG = "p";
    public static final String BAND_SIZE_FLAG = "b";

    private int bandSize = 0;

    @Override public ParamSet getParams() {
        return super.getParams().add(PENALTY_FLAG, penalty).add(BAND_SIZE_FLAG, bandSize);
    }

    @Override public void setParams(final ParamSet param) {
        ParamHandler.setParam(param, PENALTY_FLAG, this::setPenalty, Double.class);
        ParamHandler.setParam(param, BAND_SIZE_FLAG, this::setBandSize, Integer.class);
    }

    public int getBandSize() {
        return bandSize;
    }

    public void setBandSize(final int bandSize) {
        this.bandSize = bandSize;
    }
}
