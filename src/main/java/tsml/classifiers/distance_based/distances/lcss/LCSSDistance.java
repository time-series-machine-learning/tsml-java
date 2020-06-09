package tsml.classifiers.distance_based.distances.lcss;

import tsml.classifiers.distance_based.distances.BaseDistanceMeasure;
import tsml.classifiers.distance_based.distances.IntBasedWarpingDistanceMeasure;
import tsml.classifiers.distance_based.distances.WarpingDistanceMeasure;
import tsml.classifiers.distance_based.utils.params.ParamHandler;
import tsml.classifiers.distance_based.utils.params.ParamSet;
import utilities.Utilities;
import weka.core.Instance;
import weka.core.neighboursearch.PerformanceStats;

/**
 * LCSS distance measure.
 * <p>
 * Contributors: goastler
 */
public class LCSSDistance extends IntBasedWarpingDistanceMeasure {

    // delta === warp
    // epsilon === diff between two values before they're considered the same AKA tolerance

    private double epsilon = 0.01;

    public static String getEpsilonFlag() {
        return "e";
    }

    public static String getDeltaFlag() {
        return "d";
    }

    public double getEpsilon() {
        return epsilon;
    }

    public void setEpsilon(double epsilon) {
        this.epsilon = epsilon;
    }

    @Override
    public double findDistance(final double[] a, final double[] b, double limit) {

        int aLength = a.length - 1;
        int bLength = b.length - 1;

        // window should be somewhere from 0..len-1. window of 0 is ED, len-1 is Full DTW. Anything above is just
        // Full DTW
        final int windowSize = findWindowSize(aLength);

        // 22/10/19 goastler - limit LCSS such that if any value in the current window is larger than the limit then we can stop here, no point in doing the extra work
        if(limit != Double.POSITIVE_INFINITY) { // check if there's a limit set
            // if so then reverse engineer the max LCSS distance and replace the limit
            // this is just the inverse of the return value integer rounded to an LCSS distance
            limit = (int) ((1 - limit) * aLength) + 1; // todo
        }


        int[] row = new int[bLength];
        int[] prevRow = new int[bLength];
        double min = Double.POSITIVE_INFINITY;
        // top left cell of matrix will simply be the sq diff
        row[0] = approxEqual(a[0], b[0], epsilon) ? 1 : 0;
        // start and end of window
        // start at the next cell of the first row
        int start = 1;
        // end at window or bLength, whichever smallest
        int end = Math.min(bLength - 1, windowSize);
        // must set the value before and after the window to inf if available as the following row will use these
        // in top / left / top-left comparisons
        if(end + 1 < bLength) {
            row[end + 1] = Integer.MIN_VALUE;
        }
        // the first row is populated from the cell before
        for(int j = start; j <= end; j++) {
            final int cost;
            if(approxEqual(a[0], b[j], epsilon)) {
                cost = 1;
            } else {
                cost = row[j - 1];
            }
            row[j] = cost;
            min = Math.min(min, cost);
        }
        if(keepMatrix) {
            matrix = new int[aLength][bLength];
            System.arraycopy(row, 0, matrix[0], 0, row.length);
        }
        // early abandon if work has been done populating the first row for >1 entry
        if(end > start && min > limit) {
            return Double.POSITIVE_INFINITY;
        }
        for(int i = 1; i < aLength; i++) {
            // Swap current and prevRow arrays. We'll just overwrite the new row.
            {
                int[] temp = prevRow;
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
                row[start - 1] = Integer.MIN_VALUE;
            }
            if(end + 1 < bLength) {
                row[end + 1] = Integer.MIN_VALUE;
            }
            // if assessing the left most column then only top is the option - not left or left-top
            if(start == 0) {
                final int cost;
                if(approxEqual(a[i], b[start], epsilon)) {
                    cost = 1;
                } else {
                    cost = prevRow[start];
                }
                row[start] = cost;
                min = Math.min(min, cost);
                // shift to next cell
                start++;
            }
            for(int j = start; j <= end; j++) {
                final int cost;
                final int topLeft = prevRow[j - 1];
                if(approxEqual(a[i], b[j], epsilon)) {
                    cost = topLeft + 1;
                } else {
                    final int top = prevRow[j];
                    final int left = row[j - 1];
                    // max of top / left / top left
                    cost = Math.max(top, Math.max(left, topLeft));
                }
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
        return 1d - (double) row[bLength - 1] / aLength;
    }

    @Override
    public ParamSet getParams() {
        return null;
//        return super.getParams().add(getEpsilonFlag(), epsilon).add(getDeltaFlag(), delta);
    }

    @Override
    public void setParams(final ParamSet param) {
        ParamHandler.setParam(param, getEpsilonFlag(), this::setEpsilon, Double.class);
//        ParamHandler.setParam(param, getDeltaFlag(), this::setDelta, Integer.class);
    }

    public static boolean approxEqual(double a, double b, double epsilon) {
        return Math.abs(a - b) <= epsilon;
    }
}
