package tsml.classifiers.distance_based.distances.wdtw;

import tsml.classifiers.distance_based.distances.DoubleMatrixBasedDistanceMeasure;
import tsml.classifiers.distance_based.distances.WarpingParameter;
import tsml.classifiers.distance_based.utils.collections.params.ParamHandlerUtils;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import weka.core.Instance;

/**
 * WDTW distance measure.
 * <p>
 * Contributors: goastler
 */
public class WDTWDistance
    extends DoubleMatrixBasedDistanceMeasure implements WDTW {

    private double g = 0.05;
    private double[] weightVector = new double[0];
    private final WarpingParameter warpingParameter = new WarpingParameter();

    @Override
    public double getG() {
        return g;
    }

    @Override
    public void setG(double g) {
        this.g = g;
    }

    @Override
    public double findDistance(final Instance a, final Instance b, final double limit) {

        int aLength = a.numAttributes() - 1;
        int bLength = b.numAttributes() - 1;

        // generate weights
        if(aLength != weightVector.length) {
            final double halfLength = (double) aLength / 2;
            weightVector = new double[aLength];
            for(int i = 0; i < aLength; i++) {
                weightVector[i] = 1d / (1d + Math.exp(-g * (i - halfLength)));
            }
        }

        final boolean generateDistanceMatrix = isGenerateDistanceMatrix();
        final double[][] matrix = generateDistanceMatrix ? new double[aLength][bLength] : null;
        setDistanceMatrix(matrix);

        // window should be somewhere from 0..len-1. window of 0 is ED, len-1 is Full DTW. Anything above is just
        // Full DTW
        final int windowSize = findWindowSize(aLength);

        double[] row = new double[bLength];
        double[] prevRow = new double[bLength];
        // top left cell of matrix will simply be the sq diff
        double min = weightVector[0] * Math.pow(a.value(0) - b.value(0), 2);
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
            double cost = row[j - 1] + weightVector[j] * Math.pow(a.value(0) - b.value(j), 2);
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
                    prevRow[start] + weightVector[Math.abs(i - start)] * Math.pow(a.value(i) - b.value(start), 2);
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
                    Math.min(top, Math.min(left, topLeft)) + weightVector[Math.abs(i - j)] * Math.pow(a.value(i) - b.value(j), 2);
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
        return row[bLength - 1];
    }

    @Override
    public ParamSet getParams() {
        return super.getParams().add(WDTW.G_FLAG, g);
    }

    @Override
    public void setParams(final ParamSet param) throws Exception {
        super.setParams(param);
        ParamHandlerUtils.setParam(param, WDTW.G_FLAG, this::setG, Double.class);
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
