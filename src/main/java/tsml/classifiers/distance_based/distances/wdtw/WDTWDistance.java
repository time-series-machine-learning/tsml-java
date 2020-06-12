package tsml.classifiers.distance_based.distances.wdtw;

import tsml.classifiers.distance_based.distances.DoubleBasedWarpingDistanceMeasure;
import tsml.classifiers.distance_based.utils.params.ParamHandler;
import tsml.classifiers.distance_based.utils.params.ParamSet;

/**
 * WDTW distance measure.
 * <p>
 * Contributors: goastler
 */
public class WDTWDistance
    extends DoubleBasedWarpingDistanceMeasure implements WDTW {

    private double g = 0.05;
    private double[] weightVector = new double[0];

    @Override
    public double getG() {
        return g;
    }

    @Override
    public void setG(double g) {
        this.g = g;
    }

    @Override
    public double findDistance(final double[] a, final double[] b, final double limit) {

        int aLength = a.length - 1;
        int bLength = b.length - 1;

        // generate weights
        if(aLength != weightVector.length) {
            final double halfLength = (double) aLength / 2;
            weightVector = new double[aLength];
            for(int i = 0; i < aLength; i++) {
                weightVector[i] = 1d / (1d + Math.exp(-g * (i - halfLength)));
            }
        }

        // window should be somewhere from 0..len-1. window of 0 is ED, len-1 is Full DTW. Anything above is just
        // Full DTW
        final int windowSize = findWindowSize(aLength);

        double[] row = new double[bLength];
        double[] prevRow = new double[bLength];
        // top left cell of matrix will simply be the sq diff
        double min = weightVector[0] * Math.pow(a[0] - b[0], 2);
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
            double cost = row[j - 1] + weightVector[j] * Math.pow(a[0] - b[j], 2);
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
                final double cost =
                    prevRow[start] + weightVector[Math.abs(i - start)] * Math.pow(a[i] - b[start], 2);
                row[start] = cost;
                min = Math.min(min, cost);
                // shift to next cell
                start++;
            }
            for(int j = start; j <= end; j++) {
                // compute squared distance of feature vectors
                final double topLeft = prevRow[j - 1];
                final double left = row[j - 1];
                final double top = prevRow[j];
                final double cost =
                    Math.min(top, Math.min(left, topLeft)) + weightVector[Math.abs(i - j)] * Math.pow(a[i] - b[j], 2);
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
        return super.getParams().add(WDTW.getGFlag(), g);
    }

    @Override
    public void setParams(final ParamSet param) {
        ParamHandler.setParam(param, WDTW.getGFlag(), this::setG, Double.class);
    }
}
