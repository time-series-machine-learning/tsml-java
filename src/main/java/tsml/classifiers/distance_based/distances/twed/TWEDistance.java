package tsml.classifiers.distance_based.distances.twed;

import tsml.classifiers.distance_based.distances.BaseDistanceMeasure;
import tsml.classifiers.distance_based.utils.collections.params.ParamHandlerUtils;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import tsml.data_containers.TimeSeries;
import tsml.data_containers.TimeSeriesInstance;
import weka.core.Instance;

import static utilities.ArrayUtilities.*;

/**
 * TWED distance measure.
 * <p>
 * Contributors: goastler
 */
public class TWEDistance
    extends BaseDistanceMeasure {

    private double lambda;
    private double nu;

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

        final int aLength = a.getMaxLength();
        final int bLength = b.getMaxLength();

        final boolean generateDistanceMatrix = isGenerateDistanceMatrix();
        final double[][] matrix = generateDistanceMatrix ? new double[aLength][bLength] : null;
        setDistanceMatrix(matrix);

        final int windowSize = aLength + 1;

        double[] jCosts = new double[bLength + 1];
        double[] row = new double[bLength + 1];
        double[] prevRow = new double[bLength + 1];
        double dist, htrans, top, left, topLeft, cost, iCost;
        // border of the cost matrix initialization
        // top left is already 0 so don't bother checking for early abandon
        row[0] = 0;
        jCosts[1] = cellCost(b, 0);
        row[1] = jCosts[1];
        // start at the next cell
        int start = 2;
        // end at window or bLength, whichever smallest
        int end = Math.min(bLength, windowSize);
        // must set the value before and after the window to inf if available as the following row will use these
        // in top / left / top-left comparisons
        if(end + 1 < bLength + 1) {
            row[end + 1] = Double.POSITIVE_INFINITY;
        }
        for(int j = start; j <= end; j++) {
            //CHANGE AJB 8/1/16: Only use power of 2 for speed up,
            cost = cost(b, j - 2, b, j - 1);
            jCosts[j] = cost;
            row[j] = row[j - 1] + jCosts[j];
        }
        if(generateDistanceMatrix) {
            System.arraycopy(row, 0, matrix[0], 0, row.length);
        }
        {
            double[] tmp = row;
            row = prevRow;
            prevRow = tmp;
        }
        // start and end of window
        start = Math.max(0, 1 - windowSize);
        end = Math.min(bLength, 1 + windowSize);
        // must set the value before and after the window to inf if available as the following row will use these
        // in top / left / top-left comparisons
        if(start - 1 >= 0) {
            row[start - 1] = Double.POSITIVE_INFINITY;
        }
        if(end + 1 < bLength + 1) {
            row[end + 1] = Double.POSITIVE_INFINITY;
        }
        iCost = cellCost(a, 0);
        double min = Double.POSITIVE_INFINITY;
        if(start == 0) {
            row[0] = prevRow[0] + iCost;
            min = row[0];
            start++;
        }
        for(int j = start; j <= end; j++) {
            dist = cost(a, 0, b, j - 1);
            htrans = Math.abs((1 - j));
            left = prevRow[j - 1] + nu * htrans + dist;
            top = iCost + prevRow[j] + lambda + nu;
            topLeft = jCosts[j] + row[j - 1] + lambda + nu;
            cost = Math.min(left, Math.min(top, topLeft));
            row[j] = cost;
            min = Math.min(min, cost);
        }
        if(generateDistanceMatrix) {
            System.arraycopy(row, 0, matrix[1], 0, row.length);
        }
        if(end > start && min > limit) {
            return Double.POSITIVE_INFINITY;
        }
        for(int i = 2; i <= aLength; i++) {
            {
                double[] tmp = row;
                row = prevRow;
                prevRow = tmp;
            }
            min = Double.POSITIVE_INFINITY;
            // start and end of window
            start = Math.max(0, 1 - windowSize);
            end = Math.min(bLength, 1 + windowSize);
            // must set the value before and after the window to inf if available as the following row will use these
            // in top / left / top-left comparisons
            if(start - 1 >= 0) {
                row[start - 1] = Double.POSITIVE_INFINITY;
            }
            if(end + 1 < bLength + 1) {
                row[end + 1] = Double.POSITIVE_INFINITY;
            }
            iCost = cost(a, i - 2, a, i - 1);
            if(start == 0) {
                cost = prevRow[0] + iCost;
                row[0] = cost;
                min = Math.min(min, cost);
                start++;
            }
            if(start == 1) {
                dist = cost(a, i - 1, b, 0);
                htrans = i - 1;
                left = prevRow[0] + nu * htrans + dist;
                top = iCost + prevRow[1] + lambda + nu;
                topLeft = jCosts[1] + row[0] + lambda + nu;
                cost = Math.min(left, Math.min(top, topLeft));
                row[1] = cost;
                min = Math.min(min, cost);
                start++;
            }
            for(int j = start; j <= end; j++) {
                dist = cost(a, i - 1, b, j - 1) + cost(a, i - 2, b, j - 2);
                htrans = Math.abs(i - j) * 2;
                left = prevRow[j - 1] + nu * htrans + dist;
                top = iCost + prevRow[j] + lambda + nu;
                topLeft = jCosts[j] + row[j - 1] + lambda + nu;
                cost = Math.min(left, Math.min(top, topLeft));
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

        return row[bLength];

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
