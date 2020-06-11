package tsml.classifiers.distance_based.distances.msm;

import tsml.classifiers.distance_based.distances.BaseDistanceMeasure;
import tsml.classifiers.distance_based.distances.DoubleBasedWarpingDistanceMeasure;
import tsml.classifiers.distance_based.utils.params.ParamHandler;
import tsml.classifiers.distance_based.utils.params.ParamSet;
import utilities.Utilities;
import weka.core.Instance;
import weka.core.neighboursearch.PerformanceStats;

/**
 * MSM distance measure.
 * <p>
 * Contributors: goastler
 */
public class MSMDistance
    extends DoubleBasedWarpingDistanceMeasure {


    private double cost = 1;

    public MSMDistance() {

    }

    public static String getCostFlag() {
        return "c";
    }

    public double getCost() {
        return cost;
    }

    public void setCost(double cost) {
        this.cost = cost;
    }

    private double findCost(double newPoint, double x, double y) {
        double dist = 0;

        if(((x <= newPoint) && (newPoint <= y)) ||
            ((y <= newPoint) && (newPoint <= x))) {
            dist = cost;
        } else {
            dist = cost + Math.min(Math.abs(newPoint - x), Math.abs(newPoint - y));
        }

        return dist;
    }

//    @Override
//    public double distance(final Instance a,
//        final Instance b,
//        final double limit) {
//
//        checkData(a, b);
//
//        int aLength = a.numAttributes() - 1;
//        int bLength = b.numAttributes() - 1;
//
//        double[][] cost = new double[aLength][bLength];
//
//        // Initialization
//        cost[0][0] = Math.abs(a.value(0) - b.value(0));
//        for(int i = 1; i < aLength; i++) {
//            cost[i][0] = cost[i - 1][0] + findCost(a.value(i), a.value(i - 1), b.value(0));
//        }
//        for(int i = 1; i < bLength; i++) {
//            cost[0][i] = cost[0][i - 1] + findCost(b.value(i), a.value(0), b.value(i - 1));
//        }
//
//        // Main Loop
//        double min;
//        for(int i = 1; i < aLength; i++) {
//            min = limit;
//            for(int j = 1; j < bLength; j++) {
//                double d1, d2, d3;
//                d1 = cost[i - 1][j - 1] + Math.abs(a.value(i) - b.value(j));
//                d2 = cost[i - 1][j] + findCost(a.value(i), a.value(i - 1), b.value(j));
//                d3 = cost[i][j - 1] + findCost(b.value(j), a.value(i), b.value(j - 1));
//                cost[i][j] = Math.min(d1, Math.min(d2, d3));
//
//                if(cost[i][j] >= limit) {
//                    cost[i][j] = Double.POSITIVE_INFINITY;
//                }
//
//                if(cost[i][j] < min) {
//                    min = cost[i][j];
//                }
//            }
//            if(min >= limit) {
//                return Double.POSITIVE_INFINITY;
//            }
//        }
//        // Output
//        return cost[aLength - 1][bLength - 1];
//    }

    @Override
    public double findDistance(double[] a, double[] b, final double limit) {

        int aLength = a.length - 1;
        int bLength = b.length - 1;

        // window should be somewhere from 0..len-1. window of 0 is ED, len-1 is Full DTW. Anything above is just
        // Full DTW
        final int windowSize = findWindowSize(aLength);

        double[] row = new double[bLength];
        double[] prevRow = new double[bLength];
        // top left cell of matrix will simply be the sq diff
        double min = Math.abs(a[0] - b[0]);
        row[0] = min;
        // start and end of window
        // start at the next cell of the first row
        int start = 1;
        // end at window or bLength, whichever smallest
        int end = Math.min(bLength - 1, windowSize);
        // must set the value before and after the window to inf if available as the following row will use these
        // in top / left / top-left comparisons
        if(end + 1 < bLength) {
            row[end + 1] = Double.POSITIVE_INFINITY;
        }
        // the first row is populated from the sq diff + the cell before
        for(int j = start; j <= end; j++) {
            double cost = row[j - 1] + findCost(b[j], a[0], b[j - 1]);
            row[j] = cost;
            min = Math.min(min, cost);
        }
        if(keepMatrix) {
            matrix = new double[aLength][bLength];
            System.arraycopy(row, 0, matrix[0], 0, row.length);
        }
        // early abandon if work has been done populating the first row for >1 entry
        if(min > limit) {
            return Double.POSITIVE_INFINITY;
        }
        for(int i = 1; i < aLength; i++) {
            // Swap current and prevRow arrays. We'll just overwrite the new row.
            {
                double[] temp = prevRow;
                prevRow = row;
                row = temp;
            }
            // reset the insideLimit var each row. if all values for a row are above the limit then early abandon
            min = Double.POSITIVE_INFINITY;
            // start and end of window
            start = Math.max(0, i - windowSize);
            end = Math.min(bLength - 1, i + windowSize);
            // must set the value before and after the window to inf if available as the following row will use these
            // in top / left / top-left comparisons
            if(start - 1 >= 0) {
                row[start - 1] = Double.POSITIVE_INFINITY;
            }
            if(end + 1 < bLength) {
                row[end + 1] = Double.POSITIVE_INFINITY;
            }
            // if assessing the left most column then only top is the option - not left or left-top
            if(start == 0) {
                final double cost = prevRow[start] + findCost(a[i], a[i - 1], b[start]);
                row[start] = cost;
                min = Math.min(min, cost);
                // shift to next cell
                start++;
            }
            for(int j = start; j <= end; j++) {
                // compute squared distance of feature vectors
                final double topLeft = prevRow[j - 1] + Math.abs(a[i] - b[j]);
                final double top = prevRow[j] + findCost(a[i], a[i - 1], b[j]);
                final double left = row[j - 1] + findCost(b[j], a[i], b[j - 1]);
                final double cost = Math.min(top, Math.min(left, topLeft));

                row[j] = cost;
                min = Math.min(min, cost);
            }
            if(keepMatrix) {
                System.arraycopy(row, 0, matrix[i], 0, row.length);
            }
            if(min > limit) {
                return Double.POSITIVE_INFINITY;
            }
        }
        //Find the minimum distance at the end points, within the warping window.
        return row[bLength - 1];

    }

    @Override
    public ParamSet getParams() {
        return super.getParams().add(getCostFlag(), cost);
    }

    @Override
    public void setParams(final ParamSet param) {
        ParamHandler.setParam(param, getCostFlag(), this::setCost, Double.class);
    }
}
