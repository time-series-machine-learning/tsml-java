package tsml.classifiers.distance_based.distances.wdtw;

import tsml.classifiers.distance_based.distances.MatrixBasedDistanceMeasure;
import tsml.classifiers.distance_based.utils.collections.params.ParamHandlerUtils;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import tsml.data_containers.TimeSeriesInstance;

import java.util.Arrays;

import static tsml.classifiers.distance_based.distances.dtw.DTWDistance.cost;

/**
 * WDTW distance measure.
 * <p>
 * Contributors: goastler
 */
public class WDTWDistance
    extends MatrixBasedDistanceMeasure implements WDTW {

    private double g = 0.05;
    private double[] weights = new double[0];

    @Override
    public double getG() {
        return g;
    }

    @Override
    public void setG(double g) {
        this.g = g;
    }
    
    private void generateWeights(int length) {
        if(weights.length < length) {
            final double halfLength = (double) length / 2;
            double[] oldWeights = weights;
            weights = new double[length];
            System.arraycopy(oldWeights, 0, weights, 0, oldWeights.length);
            for(int i = oldWeights.length; i < length; i++) {
                weights[i] = 1d / (1d + Math.exp(-g * (i - halfLength)));
            }
        }
    }

    @Override
    public double distance(final TimeSeriesInstance a, final TimeSeriesInstance b, final double limit) {
        // collect info
        checkData(a, b, limit);
        final int aLength = a.getMaxLength();
        final int bLength = b.getMaxLength();
        final double lengthRatio = (double) bLength / aLength;
        final double windowSize = 1 * bLength;
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
        // generate weights
        generateWeights(Math.max(aLength, bLength));
        // process top left sqaure of mat
        double min = weights[0] * cost(a, 0, b, 0);
        row[0] = min;
        // compute the first row
        for(int j = start; j <= end; j++) {
            double cost = row[j - 1] + weights[j] * cost(a, 0, b, j);
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
                final double cost = prevRow[start] + weights[Math.abs(i - start)] * cost(a, i, b, start);
                row[start++] = cost;
                min = Math.min(min, cost);
            }
            // compute the distance for each cell in the row
            for(int j = start; j <= end; j++) {
                final double cost = Math.min(prevRow[j], Math.min(row[j - 1], prevRow[j - 1])) + weights[Math.abs(i - j)] * cost(a, i, b, j);
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
        return super.getParams().add(WDTW.G_FLAG, g);
    }

    @Override
    public void setParams(final ParamSet param) throws Exception {
        super.setParams(param);
        ParamHandlerUtils.setParam(param, WDTW.G_FLAG, this::setG);
    }
}
