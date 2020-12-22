package tsml.classifiers.distance_based.distances.twed;

import tsml.classifiers.distance_based.distances.MatrixBasedDistanceMeasure;
import tsml.classifiers.distance_based.utils.collections.params.ParamHandlerUtils;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import tsml.data_containers.TimeSeriesInstance;

import java.util.Arrays;

import static utilities.ArrayUtilities.*;

/**
 * TWED distance measure.
 * <p>
 * Contributors: goastler
 */
public class TWEDistance
    extends MatrixBasedDistanceMeasure {

    private double lambda = 1;
    private double nu = 1;

    public static final String NU_FLAG = "n";
    public static final String LAMBDA_FLAG = "l";

    private double cost(final TimeSeriesInstance a, final int aIndex, final TimeSeriesInstance b, final int bIndex) {
        final double[] aSlice = a.getVSliceArray(aIndex);
        final double[] bSlice = b.getVSliceArray(bIndex);
        final double[] result = subtract(aSlice, bSlice);
        pow(result, 2);
        return sum(result);
    }
    
    private double cellCost(final TimeSeriesInstance a, final int aIndex) {
        final double[] aSlice = a.getVSliceArray(aIndex);
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
        final double windowSize = 1d * bLength + 1; // +1 as the matrix is padded with 1 row and 1 col
        // start and end of window
        int i = 0; // start at first row
        double mid = i * lengthRatio;
        int start = 2; // start at 2 because 0th and 1st element is filled directly below
        int end = (int) Math.min(bLength, Math.ceil(mid + windowSize)); // +1 as matrix padded by 1 row and 1 col
        int prevEnd;
        // setup matrix and rows
        final boolean generateDistanceMatrix = isGenerateDistanceMatrix();
        final double[][] matrix;
        double[] row;
        double[] prevRow = null;
        double[] jCosts = new double[bLength + 1];
        if(generateDistanceMatrix) {
            matrix = new double[aLength + 1][bLength + 1];
            for(double[] array : matrix) Arrays.fill(array, Double.POSITIVE_INFINITY);
            row = matrix[0];
        } else {
            matrix = null;
            row = new double[bLength + 1];
            prevRow = new double[bLength + 1];
        }
        setDistanceMatrix(matrix);
        // border of the cost matrix initialization
        row[0] = 0;
        jCosts[1] = cellCost(b, 0);
        row[1] = jCosts[1];
        // compute the first padded row
        for(int j = start; j <= end; j++) {
            //CHANGE AJB 8/1/16: Only use power of 2 for speed up
            final double cost = cost(b, j - 2, b, j - 1);
            row[j] = row[j - 1] + cost;
            jCosts[j] = cost;
        }
        // compute first row
        i++; // make i==1, i.e. point to the first row. The row before is the padding row
        prevEnd = end;
        mid = i * lengthRatio;
        start = (int) Math.max(0, Math.floor(mid - windowSize)); // +1 as start from the second cell - first cell filled manually below
        end = (int) Math.min(bLength, Math.ceil(mid + windowSize)); // +1 as matrix padded with 1 row and 1 col
        double iCost = cellCost(a, 0);
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
        // compute first cell in row
        row[start++] = iCost;
        double min = iCost;
        // compute remaining cells in the first row
        for(int j = start; j <= end; j++) {
            final double dist = cost(a, i - 1, b, j - 1);
            final double htrans = Math.abs((i - j));
            final double left = prevRow[j - 1] + nu * htrans + dist;
            final double top = iCost + prevRow[j] + lambda + nu;
            final double topLeft = jCosts[j] + row[j - 1] + lambda + nu;
            final double cost = Math.min(left, Math.min(top, topLeft));
            row[j] = cost;
            min = Math.min(min, cost);
        }
        if(min > limit) return Double.POSITIVE_INFINITY; // quit if beyond limit
        // process remaining rows
        for(i = 2; i < aLength + 1; i++) {
            // reset min for the row
            min = Double.POSITIVE_INFINITY;
            // start, end and mid of window
            prevEnd = end;
            mid = i * lengthRatio;
            start = (int) Math.max(0, Math.floor(mid - windowSize));
            end = (int) Math.min(bLength, Math.ceil(mid + windowSize)); // +1 as matrix padded with 1 row and 1 col
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
            // fill any jCosts which have not yet been visited
            for(int j = prevEnd + 1; j <= end; j++) {
                jCosts[j] = cost(b, j - 2, b, j - 1);
            }
            // the ith cost for this row
            iCost = cost(a, i - 2, a, i - 1);
            // if assessing the left most column then only mapping option is top - not left or topleft
            if(start == 0) {
                final double cost = prevRow[start] + iCost;
                row[start++] = cost;
                min = Math.min(min, cost);
            }
            if(start == 1) {
                final double dist = cost(a, i - 1, b, 0);
                final double htrans = i - 1;
                final double left = prevRow[0] + nu * htrans + dist;
                final double top = iCost + prevRow[1] + lambda + nu;
                final double topLeft = jCosts[1] + row[0] + lambda + nu;
                final double cost = Math.min(left, Math.min(top, topLeft));
                row[start++] = cost;
                min = Math.min(min, cost);
            }
            // compute the distance for each cell in the row
            for(int j = start; j <= end; j++) {
                final double dist = cost(a, i - 1, b, j - 1) + cost(a, i - 2, b, j - 2);
                final double htrans = Math.abs(i - j) * 2;
                final double left = prevRow[j - 1] + nu * htrans + dist;
                final double top = iCost + prevRow[j] + lambda + nu;
                final double topLeft = jCosts[j] + row[j - 1] + lambda + nu;
                final double cost = Math.min(left, Math.min(top, topLeft));
                row[j] = cost;
                min = Math.min(min, cost);
            }
            if(min > limit) return Double.POSITIVE_INFINITY; // quit if beyond limit
        }
        // last value in the current row is the distance
        return row[row.length - 1];
    }

    public double getLambda() {
        return lambda;
    }

    public void setLambda(double lambda) {
        this.lambda = lambda;
    }

    public double getNu() {
        return nu;
    }

    public void setNu(double nu) {
        this.nu = nu;
    }

    @Override
    public ParamSet getParams() {
        return super.getParams().add(NU_FLAG, nu).add(LAMBDA_FLAG, lambda);
    }

    @Override
    public void setParams(final ParamSet param) throws Exception {
        super.setParams(param);
        ParamHandlerUtils.setParam(param, NU_FLAG, this::setNu);
        ParamHandlerUtils.setParam(param, LAMBDA_FLAG, this::setLambda);
    }

}
