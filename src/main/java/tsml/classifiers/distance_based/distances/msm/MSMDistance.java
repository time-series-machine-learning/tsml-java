package tsml.classifiers.distance_based.distances.msm;

import tsml.classifiers.distance_based.distances.BaseDistanceMeasure;
import tsml.classifiers.distance_based.utils.collections.params.ParamHandlerUtils;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import tsml.data_containers.TimeSeries;
import tsml.data_containers.TimeSeriesInstance;
import weka.core.Instance;

import static utilities.ArrayUtilities.*;

/**
 * MSM distance measure.
 * <p>
 * Contributors: goastler
 */
public class MSMDistance
    extends BaseDistanceMeasure {


    private double c = 1;

    public MSMDistance() {

    }

    public static final String C_FLAG = "c";

    public double getC() {
        return c;
    }

    public void setC(double c) {
        this.c = c;
    }

    /**
     * Find the cost for doing a move / split / merge for the univariate case.
     * @param newPoint
     * @param x
     * @param y
     * @return
     */
    private double findCost(double newPoint, double x, double y) {
        double dist = 0;

        if(((x <= newPoint) && (newPoint <= y)) ||
            ((y <= newPoint) && (newPoint <= x))) {
            dist = c;
        } else {
            dist = c + Math.min(Math.abs(newPoint - x), Math.abs(newPoint - y));
        }

        return dist;
    }

    /**
     * Find the cost for doing a move / split / merge on the multivariate case.
     * @param a
     * @param aIndex
     * @param b
     * @param bIndex
     * @param c
     * @param cIndex
     * @return
     */
    private double cost(final TimeSeriesInstance a, final int aIndex, final TimeSeriesInstance b, final int bIndex, final TimeSeriesInstance c, final int cIndex) {
        final double[] aSlice = a.getVSliceArray(aIndex);
        final double[] bSlice = b.getVSliceArray(bIndex);
        final double[] cSlice = c.getVSliceArray(cIndex);
        if(aSlice.length != bSlice.length || aSlice.length != cSlice.length) throw new IllegalStateException("dimension mismatch");
        double cost = 0;
        for(int i = 0; i < aSlice.length; i++) {
            cost += findCost(aSlice[i], bSlice[i], cSlice[i]);
        }
        return cost;
    }

    /**
     * Find the cost for a individual cell. This is used when there's no alignment change and values are mapping directly to another.
     * @param a
     * @param aIndex
     * @param b
     * @param bIndex
     * @return
     */
    private double directCost(final TimeSeriesInstance a, final int aIndex, final TimeSeriesInstance b, final int bIndex) {
        final double[] aSlice = a.getVSliceArray(aIndex);
        final double[] bSlice = b.getVSliceArray(bIndex);
        if(aSlice.length != bSlice.length) throw new IllegalStateException("dimension mismatch");
        final double[] result = subtract(aSlice, bSlice);
        abs(result);
        return sum(result);
    }

    @Override
    public double distance(TimeSeriesInstance a, TimeSeriesInstance b, final double limit) {

        final int aLength = a.getMaxLength();
        final int bLength = b.getMaxLength();

        final boolean generateDistanceMatrix = isGenerateDistanceMatrix();
        final double[][] matrix = generateDistanceMatrix ? new double[aLength][bLength] : null;
        setDistanceMatrix(matrix);

        final int windowSize = aLength;

        double[] row = new double[bLength];
        double[] prevRow = new double[bLength];
        // top left cell of matrix will simply be the sq diff
        double min = directCost(a, 0, b, 0);
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
            double cost = row[j - 1] + cost(b, j, a, 0, b, j - 1);
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
                final double cost = prevRow[start] + cost(a, i, a, i - 1, b, start);
                row[start] = cost;
                min = Math.min(min, cost);
                // shift to next cell
                start++;
            }
            for(int j = start; j <= end; j++) {
                // compute squared distance of feature vectors
                final double topLeft = prevRow[j - 1] + directCost(a, i, b, j);
                final double top = prevRow[j] + cost(a, i, a, i - 1, b, j);
                final double left = row[j - 1] + cost(b, j, a, i, b, j - 1);
                final double cost = Math.min(top, Math.min(left, topLeft));

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
        return super.getParams().add(C_FLAG, c);
    }

    @Override
    public void setParams(final ParamSet param) throws Exception {
        super.setParams(param);
        ParamHandlerUtils.setParam(param, C_FLAG, this::setC);
    }
}
