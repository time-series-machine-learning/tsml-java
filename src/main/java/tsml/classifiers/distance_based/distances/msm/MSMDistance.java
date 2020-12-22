package tsml.classifiers.distance_based.distances.msm;

import tsml.classifiers.distance_based.distances.MatrixBasedDistanceMeasure;
import tsml.classifiers.distance_based.utils.collections.params.ParamHandlerUtils;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import tsml.data_containers.TimeSeriesInstance;

import java.util.Arrays;
import java.util.stream.IntStream;

import static utilities.ArrayUtilities.*;

/**
 * MSM distance measure.
 * <p>
 * Contributors: goastler
 */
public class MSMDistance extends MatrixBasedDistanceMeasure {
    
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
        return IntStream.range(0, aSlice.length).mapToDouble(i -> findCost(aSlice[i], bSlice[i], cSlice[i])).sum();
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
        final double[] result = subtract(aSlice, bSlice);
        abs(result);
        return sum(result);
    }

    @Override
    public double distance(TimeSeriesInstance a, TimeSeriesInstance b, final double limit) {
        // collect info
        checkData(a, b, limit);
        final int aLength = a.getMaxLength();
        final int bLength = b.getMaxLength();
        final double lengthRatio = (double) bLength / aLength;
        final double windowSize = 1d * bLength;
        // start and end of window
        int start = 1; // start at 1 because 0th element is filled directly below
        double mid;
        int end =  (int) Math.min(bLength - 1, Math.ceil(windowSize));
        int prevEnd;
        // setup matrix and rows
        final boolean generateDistanceMatrix = isGenerateDistanceMatrix();
        final double[][] matrix;
        double[] row;
        double[] prevRow = null;
        if(generateDistanceMatrix) {
            matrix = new double[aLength][bLength];
            for(double[] array : matrix) Arrays.fill(array, Double.POSITIVE_INFINITY);
            row = matrix[0];
        } else {
            matrix = null;
            row = new double[bLength];
            prevRow = new double[bLength];
        }
        setDistanceMatrix(matrix);
        // process top left sqaure of mat
        double min = directCost(a, 0, b, 0);
        row[0] = min;
        // compute the first row
        for(int j = start; j <= end; j++) {
            double cost = row[j - 1] + cost(b, j, a, 0, b, j - 1);
            row[j] = cost;
            min = Math.min(min, cost);
        }
        if(min > limit) return Double.POSITIVE_INFINITY; // quit if beyond limit
        // process remaining rows
        for(int i = 1; i < aLength; i++) {
            // reset min for the row
            min = Double.POSITIVE_INFINITY;
            // start, end and mid of window
            prevEnd = end;
            mid = i * lengthRatio;
            start = (int) Math.max(0, Math.floor(mid - windowSize));
            end = (int) Math.min(bLength - 1, Math.ceil(mid + windowSize));
            // change rows
            if(generateDistanceMatrix) {
                row = matrix[i];
                prevRow = matrix[i - 1];
            } else {
                // reuse previous row
                double[] tmp = row;
                row = prevRow;
                prevRow = tmp;
                // set the top values outside of window to inf
                Arrays.fill(prevRow, prevEnd + 1, end + 1, Double.POSITIVE_INFINITY);
                // set the value left of the window to inf
                if(start > 0) row[start - 1] = Double.POSITIVE_INFINITY;
            }
            // if assessing the left most column then only mapping option is top - not left or topleft
            if(start == 0) {
                final double cost = prevRow[start] + cost(a, i, a, i - 1, b, start);
                row[start++] = cost;
                min = Math.min(min, cost);
            }
            // compute the distance for each cell in the row
            for(int j = start; j <= end; j++) {
                final double topLeft = prevRow[j - 1] + directCost(a, i, b, j);
                final double top = prevRow[j] + cost(a, i, a, i - 1, b, j);
                final double left = row[j - 1] + cost(b, j, a, i, b, j - 1);
                final double cost = Math.min(top, Math.min(left, topLeft));
                row[j] = cost;
                min = Math.min(min, cost);
            }
            if(min > limit) return Double.POSITIVE_INFINITY; // quit if beyond limit
        }
        // last value in the current row is the distance
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
