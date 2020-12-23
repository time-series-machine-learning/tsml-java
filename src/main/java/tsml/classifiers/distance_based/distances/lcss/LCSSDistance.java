package tsml.classifiers.distance_based.distances.lcss;

import tsml.classifiers.distance_based.distances.MatrixBasedDistanceMeasure;
import tsml.classifiers.distance_based.utils.collections.params.ParamHandlerUtils;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import tsml.data_containers.TimeSeriesInstance;

import java.util.Arrays;

import static tsml.classifiers.distance_based.distances.dtw.DTW.WINDOW_SIZE_FLAG;
import static utilities.ArrayUtilities.*;

/**
 * LCSS distance measure.
 * <p>
 * Contributors: goastler
 */
public class LCSSDistance extends MatrixBasedDistanceMeasure {
    
    // delta === warp
    // epsilon === diff between two values before they're considered the same AKA tolerance
    
    private double epsilon = 0.01;
    private double windowSize = 1;

    public static final String EPSILON_FLAG = "e";

    public double getEpsilon() {
        return epsilon;
    }

    public void setEpsilon(double epsilon) {
        this.epsilon = epsilon;
    }

    private boolean approxEqual(TimeSeriesInstance a, int aIndex, TimeSeriesInstance b, int bIndex) {
        for(int i = 0; i < a.getNumDimensions(); i++) {
            final Double aValue = a.get(i).get(aIndex);
            final Double bValue = b.get(i).get(bIndex);
            if(Math.abs(aValue - bValue) > epsilon) {
                return false;
            }
        }
        return true;
    }

    @Override
    public double distance(final TimeSeriesInstance a, final TimeSeriesInstance b, double limit) {
        // collect info
        checkData(a, b, limit);
        final int aLength = a.getMaxLength();
        final int bLength = b.getMaxLength();
        final double lengthRatio = (double) bLength / aLength;
        final double windowSize = this.windowSize * bLength;
        // 22/10/19 goastler - limit LCSS such that if any value in the current window is larger than the limit then we can stop here, no point in doing the extra work
        if(limit != Double.POSITIVE_INFINITY) { // check if there's a limit set
            // if so then reverse engineer the max LCSS distance and replace the limit
            // this is just the inverse of the return value integer rounded to an LCSS distance
            limit = (1 - limit) * Math.min(aLength, bLength);
            // is potentially slightly too low, causing *early* early abandon
        }
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
            for(double[] array : matrix) Arrays.fill(array, Double.NEGATIVE_INFINITY);
            row = matrix[0];
        } else {
            matrix = null;
            row = new double[bLength];
            prevRow = new double[bLength];
        }
        setDistanceMatrix(matrix);
        // process top left sqaure of mat
        double min, cost;
        row[0] = min = cost = approxEqual(a, 0, b, 0) ? 1 : 0;
        // compute the first row
        for(int j = start; j <= end; j++) {
            if(approxEqual(a, 0, b, j)) {
                cost = 1;
            } else {
                cost = row[j - 1];
            }
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
                Arrays.fill(prevRow, prevEnd + 1, end + 1, Double.NEGATIVE_INFINITY);
                // set the value left of the window to inf
                if(start > 0) row[start - 1] = Double.NEGATIVE_INFINITY;
            }
            // if assessing the left most column then only mapping option is top - not left or topleft
            if(start == 0) {
                if(approxEqual(a, i, b, start)) {
                    cost = 1;
                } else {
                    cost = prevRow[start];
                }
                row[start++] = cost;
                min = Math.min(min, cost);
            }
            // compute the distance for each cell in the row
            for(int j = start; j <= end; j++) {
                if(approxEqual(a, i, b, j)) {
                    cost = prevRow[j - 1] + 1;
                } else {
                    cost = Math.max(row[j - 1], Math.max(prevRow[j], prevRow[j - 1]));
                }
                row[j] = cost;
                min = Math.min(min, cost);
            }
            if(min > limit) return Double.POSITIVE_INFINITY; // quit if beyond limit
        }
        // last value in the current row is the distance
        return 1d - row[bLength - 1] / Math.min(aLength, bLength);
    }

    @Override
    public ParamSet getParams() {
        return super.getParams().add(WINDOW_SIZE_FLAG, windowSize).add(EPSILON_FLAG, epsilon);
    }

    @Override
    public void setParams(final ParamSet param) throws Exception {
        ParamHandlerUtils.setParam(param, EPSILON_FLAG, this::setEpsilon, Double::parseDouble);
        ParamHandlerUtils.setParam(param, WINDOW_SIZE_FLAG, this::setWindowSize, Double::parseDouble);
        super.setParams(param);
    }

    public double getWindowSize() {
        return windowSize;
    }

    public void setWindowSize(final double windowSize) {
        this.windowSize = windowSize;
    }
}
