package tsml.classifiers.distance_based.distances.dtw;


import javax.rmi.CORBA.Util;
import tsml.classifiers.distance_based.distances.BaseDistanceMeasure;
import tsml.classifiers.distance_based.utils.instance.ExposedDenseInstance;
import tsml.classifiers.distance_based.utils.params.ParamHandler;
import tsml.classifiers.distance_based.utils.params.ParamSet;
import utilities.Utilities;
import weka.core.Instance;

/**
 * DTW distance measure.
 * <p>
 * Contributors: goastler
 */
public class DTWDistance extends BaseDistanceMeasure implements DTW {

    // the distance matrix produced by the distance function
    private double[][] matrix;
    // whether to keep the distance matrix
    private boolean keepMatrix = false;
    private int windowSize = -1;
    private double windowSizePercentage = -1;
    private boolean windowSizeInPercentage = false;

    public DTWDistance() {}

    public int getWindowSize() {
        return windowSize;
    }

    @Override
    public void setWindowSizePercentage(final double percentage) {
        this.windowSizePercentage = percentage;
        windowSizeInPercentage = true;
    }

    @Override
    public double getWindowSizePercentage() {
        return windowSizePercentage;
    }

    public void setWindowSize(int windowSize) {
        this.windowSize = windowSize;
        windowSizeInPercentage = false;
    }

    public double[][] getMatrix() {
        return matrix;
    }

    protected void setMatrix(double[][] matrix) {
        if(keepMatrix) {
            this.matrix = matrix;
        }
    }

    public boolean isKeepMatrix() {
        return keepMatrix;
    }

    public void setKeepMatrix(final boolean keepMatrix) {
        this.keepMatrix = keepMatrix;
    }

    @Override
    public void clean() {
        super.clean();
        cleanDistanceMatrix();
    }

    public void cleanDistanceMatrix() {
        matrix = null;
    }

    @Override
    public boolean isWindowSizeInPercentage() {
        return windowSizeInPercentage;
    }

    @Override
    public double distance(Instance ai, Instance bi, final double limit) {

        checkData(ai, bi);

        double[] a = ExposedDenseInstance.extractAttributeValuesAndClassLabel(ai);
        double[] b = ExposedDenseInstance.extractAttributeValuesAndClassLabel(bi);

        int aLength = a.length - 1;
        int bLength = b.length - 1;

        // put a or first as the longest time series
        if(bLength > aLength) {
            double[] tmp = a;
            a = b;
            b = tmp;
            int tmpLength = aLength;
            aLength = bLength;
            bLength = tmpLength;
        }

        // window should be somewhere from 0..len-1. window of 0 is ED, len-1 is Full DTW. Anything above is just
        // Full DTW
        final int windowSize;
        if(windowSizeInPercentage) {
            if(windowSizePercentage > 1) {
                throw new IllegalArgumentException("window percentage size larger than 1");
            }
            if(windowSizePercentage < 0) {
                windowSize = aLength - 1;
            } else {
                windowSize = (int) (windowSizePercentage * (aLength - 1));
            }
        } else {
            if(this.windowSize > aLength - 1) {
                throw new IllegalArgumentException("window size larger than series length: " + this.windowSize + " vs"
                    + " " + (aLength - 1));
            }
            if(this.windowSize < 0) {
                windowSize = aLength - 1;
            } else {
                windowSize = this.windowSize;
            }
        }

        double[] row = new double[bLength];
        double[] prevRow = new double[bLength];
        boolean insideLimit = true;
        // top left cell of matrix will simply be the sq diff
        row[0] = Math.pow(a[0] - b[0], 2);
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
            double cost = row[j - 1] + Math.pow(a[0] - b[j], 2);
            row[j] = cost;
            if(insideLimit && cost < limit) {
                insideLimit = false;
            }
        }
        if(keepMatrix) {
            matrix = new double[aLength][bLength];
        }
        // early abandon if work has been done populating the first row for >1 entry
        if(end > start && insideLimit) {
            return Double.POSITIVE_INFINITY;
        }
        // Swap current and prevRow arrays. We'll just overwrite the new row.
        {
            double[] temp = prevRow;
            prevRow = row;
            row = temp;
        }
        for(int i = 1; i < aLength; i++) {
            // reset the insideLimit var each row. if all values for a row are above the limit then early abandon
            insideLimit = true;
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
                final double cost = prevRow[start] + Math.pow(a[i] - b[start], 2);
                row[start] = cost;
                if(insideLimit && cost < limit) {
                    insideLimit = false;
                }
                // shift to next cell
                start++;
            }
            for(int j = start; j <= end; j++) {
                // compute squared distance of feature vectors
                final double topLeft = j > 0 ? prevRow[j - 1] : Double.POSITIVE_INFINITY;
                final double left = j > 0 ? row[j - 1] : Double.POSITIVE_INFINITY;
                final double top = prevRow[j];
                final double cost = Utilities.min(top, left, topLeft) + Math.pow(a[i] - b[j], 2);

                row[j] = cost;

                if(insideLimit && cost < limit) {
                    insideLimit = false;
                }
            }
            if(keepMatrix) {
                System.arraycopy(row, 0, matrix[i], 0, row.length);
            }
            if(insideLimit) {
                return Double.POSITIVE_INFINITY;
            }
            // Swap current and prevRow arrays. We'll just overwrite the new row.
            {
                double[] temp = prevRow;
                prevRow = row;
                row = temp;
            }
        }
        //Find the minimum distance at the end points, within the warping window.
        return prevRow[bLength - 1];
    }

    @Override
    public ParamSet getParams() {
        return super.getParams().add(DTW.getWarpingWindowFlag(), getWindowSize());
    }

    @Override
    public void setParams(final ParamSet param) {
        ParamHandler.setParam(param, DTW.getWarpingWindowFlag(), this::setWindowSize, Integer.class);
    }
}
