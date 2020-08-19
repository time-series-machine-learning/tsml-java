package tsml.classifiers.distance_based.distances.lcss;

import tsml.classifiers.distance_based.distances.IntMatrixBasedDistanceMeasure;
import tsml.classifiers.distance_based.distances.WarpingParameter;
import tsml.classifiers.distance_based.utils.collections.params.ParamHandlerUtils;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import weka.core.Instance;

/**
 * LCSS distance measure.
 * <p>
 * Contributors: goastler
 */
public class LCSSDistance extends IntMatrixBasedDistanceMeasure {

    public static final String WINDOW_SIZE_FLAG = WarpingParameter.WINDOW_SIZE_FLAG;
    public static final String WINDOW_SIZE_PERCENTAGE_FLAG = WarpingParameter.WINDOW_SIZE_PERCENTAGE_FLAG;

    private final WarpingParameter warpingParameter = new WarpingParameter();
    // delta === warp
    // epsilon === diff between two values before they're considered the same AKA tolerance

    private double epsilon = 0.01;

    public static final String EPSILON_FLAG = "e";

    private static boolean approxEqual(double a, double b, double epsilon) {
        return Math.abs(a - b) <= epsilon;
    }

    public double getEpsilon() {
        return epsilon;
    }

    public void setEpsilon(double epsilon) {
        this.epsilon = epsilon;
    }

    @Override
    public double findDistance(final Instance a, final Instance b, double limit) {

        int aLength = a.numAttributes() - 1;
        int bLength = b.numAttributes() - 1;


        final boolean generateDistanceMatrix = isGenerateDistanceMatrix();
        final int[][] matrix = generateDistanceMatrix ? new int[aLength][bLength] : null;
        setDistanceMatrix(matrix);

        // window should be somewhere from 0..len-1. window of 0 is ED, len-1 is Full DTW. Anything above is just
        // Full DTW
        final int windowSize = findWindowSize(aLength);

        // 22/10/19 goastler - limit LCSS such that if any value in the current window is larger than the limit then we can stop here, no point in doing the extra work
        if(limit != Double.POSITIVE_INFINITY) { // check if there's a limit set
            // if so then reverse engineer the max LCSS distance and replace the limit
            // this is just the inverse of the return value integer rounded to an LCSS distance
            limit = (int) ((1 - limit) * aLength) + 1; // must have plus 1 due to int rounding. Otherwise the value
            // is potentially slightly too low, causing *early* early abandon
        }

        int[] row = new int[bLength];
        int[] prevRow = new int[bLength];
        // init min to top left cell
        double min = approxEqual(a.value(0), b.value(0), epsilon) ? 1 : 0;
        // top left cell of matrix will simply be the sq diff
        row[0] = (int) min;
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
            if(approxEqual(a.value(0), b.value(j), epsilon)) {
                cost = 1;
            } else {
                cost = row[j - 1];
            }
            row[j] = cost;
            min = Math.min(min, cost);
        }
        if(generateDistanceMatrix) {
            System.arraycopy(row, 0, matrix[0], 0, row.length);
        }
        // early abandon if work has been done populating the first row for >1 entry
        if(min > limit) {
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
                if(approxEqual(a.value(i), b.value(start), epsilon)) {
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
                if(approxEqual(a.value(i), b.value(j), epsilon)) {
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
            if(generateDistanceMatrix) {
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
        return super.getParams().addAll(warpingParameter.getParams()).add(EPSILON_FLAG, epsilon);
    }

    @Override
    public void setParams(final ParamSet param) throws Exception {
        ParamHandlerUtils.setParam(param, EPSILON_FLAG, this::setEpsilon, Double.class);
        warpingParameter.setParams(param);
        super.setParams(param);
    }

    public int findWindowSize(final int aLength) {
        return warpingParameter.findWindowSize(aLength);
    }

    public int getWindowSize() {
        return warpingParameter.getWindowSize();
    }

    public void setWindowSize(final int windowSize) {
        warpingParameter.setWindowSize(windowSize);
    }

    public double getWindowSizePercentage() {
        return warpingParameter.getWindowSizePercentage();
    }

    public void setWindowSizePercentage(final double windowSizePercentage) {
        warpingParameter.setWindowSizePercentage(windowSizePercentage);
    }

    public boolean isWindowSizeInPercentage() {
        return warpingParameter.isWindowSizeInPercentage();
    }
}
