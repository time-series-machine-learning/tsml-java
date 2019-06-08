package distances.time_domain.wdtw;

import distances.DistanceMeasure;
import evaluation.tuning.ParameterSpace;
import utilities.Utilities;
import weka.core.Instance;

public class Wdtw
    extends DistanceMeasure {

    public double getWeight() {
        return weight;
    }

    public void setWeight(double weight) {
        this.weight = weight;
    }

    public Wdtw(double weight) {
        setWeight(weight);
    }

    public static final double DEFAULT_WEIGHT = 0.05;

    public Wdtw() {
        this(DEFAULT_WEIGHT);
    }

    private double weight; // AKA g // 0.05 to 3 perhaps

    @Override
    public double distance(Instance a,
                           Instance b,
                           final double cutOff) {

        // todo cleanup
        // todo trim memory to window by window
        // todo early abandon

        double[] timeSeriesA = Utilities.extractAttributesNoClassValue(a);
        double[] timeSeriesB = Utilities.extractAttributesNoClassValue(b);
        int seriesLength = timeSeriesA.length;
        double[] weightVector = new double[seriesLength];
        double halfLength = (double) seriesLength / 2;

        for (int i = 0; i < seriesLength; i++) { // todo precompute and use in findCost func
            weightVector[i] = 1 / (1 + Math.exp(-weight * (i - halfLength)));
        }

        //create empty array
        int m = timeSeriesA.length;
        int n = timeSeriesB.length;
        double[][] distances = new double[m][n];

        //first value
        distances[0][0] = weightVector[0] * (timeSeriesA[0] - timeSeriesB[0]) * (timeSeriesA[0] - timeSeriesB[0]);

        //early abandon if first values is larger than cut off
        if (distances[0][0] > cutOff) {
            return Double.MAX_VALUE;
        }

        //top row
        for (int i = 1; i < n; i++) {
            distances[0][i] =
                distances[0][i - 1] + weightVector[i] * (timeSeriesA[0] - timeSeriesB[i]) * (timeSeriesA[0] - timeSeriesB[i]); //edited by Jay
        }

        //first column
        for (int i = 1; i < m; i++) {
            distances[i][0] =
                distances[i - 1][0] + weightVector[i] * (timeSeriesA[i] - timeSeriesB[0]) * (timeSeriesA[i] - timeSeriesB[0]); //edited by Jay
        }

        //warp rest
        double minDistance;
        for (int i = 1; i < m; i++) {
            boolean overflow = true;

            for (int j = 1; j < n; j++) {
                //calculate distances
                minDistance = Math.min(distances[i][j - 1], Math.min(distances[i - 1][j], distances[i - 1][j - 1]));
                distances[i][j] =
                    minDistance + weightVector[Math.abs(i - j)] * (timeSeriesA[i] - timeSeriesB[j]) * (timeSeriesA[i] - timeSeriesB[j]);

                if (overflow && distances[i][j] < cutOff) {
                    overflow = false; // because there's evidence that the path can continue
                }
            }

            //early abandon
            if (overflow) {
                return Double.MAX_VALUE;
            }
        }
        return distances[m - 1][n - 1];
    }


    public static final String WEIGHT_KEY = "weight";

    @Override
    public void setOptions(String[] options) {
        for (int i = 0; i < options.length - 1; i += 2) {
            String key = options[i];
            String value = options[i + 1];
            if (key.equals(WEIGHT_KEY)) {
                setWeight(Double.parseDouble(value));
            }
        }
    }

    @Override
    public String[] getOptions() {
        return new String[] {
            WEIGHT_KEY,
            String.valueOf(weight),
            };
    }


    public static final String NAME = "WDTW";

    @Override
    public String toString() {
        return NAME;
    }

    public static ParameterSpace discreteParameterSpace() {
        double[] gValues = new double[100];
        for(int i = 0; i < gValues.length; i++) {
            gValues[i] = i / gValues.length;
        }
        ParameterSpace parameterSpace = new ParameterSpace();
        parameterSpace.addParameter(DISTANCE_MEASURE_KEY, new String[] {NAME});
        parameterSpace.addParameter(WEIGHT_KEY, gValues);
        return parameterSpace;
    }

}
