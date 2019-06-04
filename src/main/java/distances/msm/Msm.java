package distances.msm;

import distances.DistanceMeasure;
import distances.dtw.Dtw;
import utilities.ArrayUtilities;

public class Msm
    extends DistanceMeasure {

    public Msm() {
        this(DEFAULT_PENALTY);
    }


    public Msm(double penalty) {
        setPenalty(penalty);
    }

    public double getPenalty() {
        return penalty;
    }

    public void setPenalty(double penalty) {
        this.penalty = penalty;
    }

    private double penalty = 1; // AKA cost

    // todo implement warping window

    private double findCost(double new_point, double x, double y) {
        double dist = 0;

        if (((x <= new_point) && (new_point <= y)) ||
            ((y <= new_point) && (new_point <= x))) {
            dist = getPenalty();
        } else {
            dist = getPenalty() + Math.min(Math.abs(new_point - x), Math.abs(new_point - y));
        }

        return dist;
    }

    @Override
    protected double measureDistance(double[] timeSeriesA, double[] timeSeriesB, double cutOff) {
        // todo cleanup
        // todo trim memory to window by window
        // todo early abandon
        // todo can probably use DTW's implementation and just provide a different cost function

        int m, n, i, j;
        m = timeSeriesA.length;
        n = timeSeriesB.length;
        int warpingWindow = timeSeriesA.length;
//        int warpingWindow = (int) (getWarpingWindow() * timeSeriesA.length);
        warpingWindow--;
        if (warpingWindow < 0) {
            warpingWindow = 0;
        }

        double[][] cost = new double[m][n];

        // Initialization
        cost[0][0] = Math.abs(timeSeriesA[0] - timeSeriesB[0]);

        int p = Math.min(timeSeriesB.length - 1, 1 + warpingWindow);
        for (i = 1; i < m; i++) {
            if (i < p) {
                cost[i][0] = cost[i - 1][0] + findCost(timeSeriesA[i], timeSeriesA[i - 1], timeSeriesB[0]);
            } else {
                cost[i][0] = Double.POSITIVE_INFINITY;
            }
        }

        for (j = 1; j < n; j++) {
            if (j < p) {
                cost[0][j] = cost[0][j - 1] + findCost(timeSeriesB[j], timeSeriesA[0], timeSeriesB[j - 1]);
            } else {
                cost[0][j] = Double.POSITIVE_INFINITY;
            }
        }

        // Main Loop
        for (i = 1; i < m; i++) {
            int warpingWindowStart = Math.max(0, i - warpingWindow);
            int warpingWindowEnd = Math.min(timeSeriesB.length - 1, i + warpingWindow);
            for (j = 1; j < n; j++) {
                if (j < warpingWindowStart || j > warpingWindowEnd) {
                    cost[i][j] = Double.POSITIVE_INFINITY;
                } else {
                    double d1, d2, d3;
                    d1 = cost[i - 1][j - 1] + Math.abs(timeSeriesA[i] - timeSeriesB[j]);
                    d2 = cost[i - 1][j] + findCost(timeSeriesA[i], timeSeriesA[i - 1], timeSeriesB[j]);
                    d3 = cost[i][j - 1] + findCost(timeSeriesB[j], timeSeriesA[i], timeSeriesB[j - 1]);
                    cost[i][j] = Math.min(d1, Math.min(d2, d3));
                }
            }
        }

        // Output
        return cost[m - 1][n - 1];

        //        double[] rowOne = new double[timeSeriesB.length];
//        rowOne[0] = Math.abs(timeSeriesA[0] - timeSeriesB[0]);
//        double min = rowOne[0];
//        for(int columnIndex = 1; columnIndex < timeSeriesB.length; columnIndex++) {
//            rowOne[columnIndex] = rowOne[columnIndex - 1] + findCost(timeSeriesB[columnIndex], timeSeriesA[0],
//            timeSeriesB[columnIndex - 1]);
//            min = min(min, rowOne[columnIndex]);
//        }
//        if(min > cutOff) {
//            return Double.POSITIVE_INFINITY;
//        }
//        for(int rowIndex = 1; rowIndex < timeSeriesA.length; rowIndex++) {
//            double[] rowTwo = new double[timeSeriesB.length];
//            rowTwo[0] = rowOne[0] + findCost(timeSeriesA[rowIndex], timeSeriesA[rowIndex - 1], timeSeriesB[0]);
//            min = rowTwo[0];
//            for(int columnIndex = 1; columnIndex < timeSeriesB.length; columnIndex++) {
//                double timeseriesweka.classifiers.ee = rowOne[columnIndex - 1] + Math.abs(timeSeriesA[rowIndex] -
//                timeSeriesB[columnIndex]);
//                double b = rowOne[columnIndex] + findCost(timeSeriesA[rowIndex], timeSeriesA[rowIndex - 1],
//                timeSeriesB[columnIndex]);
//                double c = rowTwo[columnIndex - 1] + findCost(timeSeriesB[columnIndex], timeSeriesA[rowIndex],
//                timeSeriesB[columnIndex - 1]);
//                rowTwo[columnIndex] = min(timeseriesweka.classifiers.ee,b,c);
//                min = min(min, rowTwo[columnIndex]);
//            }
//            if(min > cutOff) {
//                return Double.POSITIVE_INFINITY;
//            }
//            rowOne = rowTwo;
//        }
//        return rowOne[rowOne.length - 1];

//        double[][] penalty = new double[timeSeriesA.length][timeSeriesB.length];
//        penalty[0][0] = Math.abs(timeSeriesA[0] - timeSeriesB[0]);
//        for(int i = 1; i < timeSeriesA.length; i++) {
//            penalty[i][0] = penalty[i - 1][0] + findCost(timeSeriesA[i], timeSeriesA[i - 1], timeSeriesB[0]);
//        }
//        for(int j = 1; j < timeSeriesA.length; j++) {
//            penalty[0][j] = penalty[0][j - 1] + findCost(timeSeriesB[j], timeSeriesA[0], timeSeriesB[j - 1]);
//        }
//        for(int i = 1; i < timeSeriesA.length; i++) {
//            for(int j = 1; j < timeSeriesB.length; j++) {
//                double timeseriesweka.classifiers.ee = penalty[i - 1][j - 1] + Math.abs(timeSeriesA[i] -
//                timeSeriesB[j]);
//                double b = penalty[i - 1][j] + findCost(timeSeriesA[i], timeSeriesA[i - 1], timeSeriesB[j]);
//                double c = penalty[i][j - 1] + findCost(timeSeriesB[j], timeSeriesA[i], timeSeriesB[j - 1]);
//                penalty[i][j] = min(timeseriesweka.classifiers.ee,b,c);
//            }
//        }
//        return penalty[timeSeriesA.length - 1][timeSeriesB.length - 1];
    }

    public static final String PENALTY_KEY = "penalty";

    public static final double DEFAULT_PENALTY = 1;

    @Override
    public void setOptions(String[] options) {
        for (int i = 0; i < options.length - 1; i += 2) {
            String key = options[i];
            String value = options[i + 1];
            if (key.equals(PENALTY_KEY)) {
                setPenalty(Double.parseDouble(value));
            }
        }
    }

    @Override
    public String[] getOptions() {
        return ArrayUtilities.concat(super.getOptions(), new String[] {
            PENALTY_KEY,
            String.valueOf(penalty)
        });
    }
}
