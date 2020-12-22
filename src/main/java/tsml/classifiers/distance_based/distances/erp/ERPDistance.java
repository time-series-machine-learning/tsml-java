package tsml.classifiers.distance_based.distances.erp;

import tsml.classifiers.distance_based.distances.MatrixBasedDistanceMeasure;
import tsml.classifiers.distance_based.utils.collections.params.ParamHandlerUtils;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import tsml.data_containers.TimeSeriesInstance;
import utilities.Utilities;

import java.util.Arrays;

import static tsml.classifiers.distance_based.distances.dtw.DTW.WINDOW_SIZE_FLAG;
import static utilities.ArrayUtilities.*;

/**
 * ERP distance measure.
 * <p>
 * Contributors: goastler
 */
public class ERPDistance extends MatrixBasedDistanceMeasure {

    public static final String G_FLAG = "g";
    private double g = 0.01;
    private double windowSize = 1;
    
    public double getG() {
        return g;
    }

    public void setG(double g) {
        this.g = g;
    }
    
    public double cost(final TimeSeriesInstance a, final int aIndex) {
        final double[] aSlice = a.getVSliceArray(aIndex);
        subtract(aSlice, g);
        pow(aSlice, 2);
        return sum(aSlice);
    }
    
    @Override
    public double distance(final TimeSeriesInstance a, final TimeSeriesInstance b, final double limit) {
        // collect info
        checkData(a, b, limit);
        final int aLength = a.getMaxLength();
        final int bLength = b.getMaxLength();
        final double lengthRatio = (double) bLength / aLength;
        final double windowSize = this.windowSize * bLength;
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
        double min = 0; // top left cell is always zero
        row[0] = min;
        // compute the first row
        for(int j = start; j <= end; j++) {
            double cost = row[j - 1] + cost(b, j);
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
                final double cost = prevRow[start] + cost(a, i);
                row[start++] = cost;
                min = Math.min(min, cost);
            }
            // compute the distance for each cell in the row
            for(int j = start; j <= end; j++) {
                final double[] v1 = a.getVSliceArray(i);
                final double[] v2 = b.getVSliceArray(j);
                final double leftPenalty = sum(pow(subtract(copy(v1), g), 2));
                final double topPenalty = sum(pow(subtract(copy(v2), g), 2));
                final double topLeftPenalty = sum(pow(subtract(v1, v2), 2));
                final double topLeft = prevRow[j - 1] + topLeftPenalty;
                final double left = row[j - 1] + topPenalty;
                final double top = prevRow[j] + leftPenalty;
                final double cost;
                if(topLeft > left && left < top) {
                    // del
                    cost = left;
                } else if(topLeft > top && top < left) {
                    // ins
                    cost = top;
                } else {
                    // match
                    cost = topLeft;
                }
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
        return super.getParams().add(WINDOW_SIZE_FLAG, windowSize).add(G_FLAG, g);
    }

    @Override
    public void setParams(final ParamSet param) throws Exception {
        super.setParams(param);
        ParamHandlerUtils.setParam(param, G_FLAG, this::setG);
        ParamHandlerUtils.setParam(param, WINDOW_SIZE_FLAG, this::setWindowSize);
    }

    public double getWindowSize() {
        return windowSize;
    }

    public void setWindowSize(final double windowSize) {
        this.windowSize = Utilities.requirePercentage(windowSize);
    }
}
