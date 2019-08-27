package timeseriesweka.classifiers.distance_based.distance_measures;

import weka.core.Instance;

public class Erp extends DistanceMeasure {

    private double penalty = 0;

    public double getPenalty() {
        return penalty;
    }

    public void setPenalty(double g) {
        this.penalty = g;
    }

    @Override
    public double measureDistance() {

        Instance a = getFirstInstance();
        Instance b = getSecondInstance();
        int aLength = a.numAttributes() - 1;
        int bLength = b.numAttributes() - 1;

        // Current and previous columns of the matrix
        double[] curr = new double[bLength];
        double[] prev = new double[bLength];

        // size of edit distance band
        // bandsize is the maximum allowed distance to the diagonal
//        int band = (int) Math.ceil(v2.getDimensionality() * bandSize);
        int band = getBandSize();
        if(band < 0) {
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

            for (int j = l;
                 j <= r;
                 j++) {
                if (Math.abs(i - j) <= band) {
                    // compute squared distance of feature vectors
                    double val1 = a.value(i);
                    double val2 = gValue;
                    double diff = (val1 - val2);
                    final double d1 = Math.sqrt(diff * diff);

                    val1 = gValue;
                    val2 = b.value(j);
                    diff = (val1 - val2);
                    final double d2 = Math.sqrt(diff * diff);

                    val1 = a.value(i);
                    val2 = b.value(j);
                    diff = (val1 - val2);
                    final double d12 = Math.sqrt(diff * diff);

                    final double dist1 = d1 * d1;
                    final double dist2 = d2 * d2;
                    final double dist12 = d12 * d12;

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
                    // steps[i][j] = step;
                } else {
                    curr[j] = Double.POSITIVE_INFINITY; // outside band
                }
            }
        }

        return Math.sqrt(curr[bLength - 1]);
    }


    public static final String PENALTY_KEY = "penalty";
    public static final String BAND_SIZE_KEY = "bandSize";

    private int bandSize = 0;

    @Override
    public String[] getOptions() {
        return new String[] {
            PENALTY_KEY,
            String.valueOf(penalty),
            BAND_SIZE_KEY,
            String.valueOf(bandSize)
        };
    }


    @Override
    public void setOption(final String key, final String value) {
        if (key.equals(PENALTY_KEY)) {
            setPenalty(Double.parseDouble(value));
        } else if(
            key.equals(BAND_SIZE_KEY)
        ) {
            setBandSize(Integer.parseInt(value));
        }
    }

    public static final String NAME = "ERP";

    @Override
    public String toString() {
        return NAME;
    }

    public int getBandSize() {
        return bandSize;
    }

    public void setBandSize(final int bandSize) {
        this.bandSize = bandSize;
    }
}
