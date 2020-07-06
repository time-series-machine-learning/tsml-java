package tsml.classifiers.distance_based.distances.twed;

import tsml.classifiers.distance_based.distances.DoubleMatrixBasedDistanceMeasure;
import tsml.classifiers.distance_based.utils.collections.params.ParamHandlerUtils;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import weka.core.Instance;

/**
 * TWED distance measure.
 * <p>
 * Contributors: goastler
 */
public class TWEDistance
    extends DoubleMatrixBasedDistanceMeasure {

    private double lambda;
    private double nu;

    public static final String NU_FLAG = "n";
    public static final String LAMBDA_FLAG = "l";

    @Override
    public double findDistance(final Instance a, final Instance b, final double limit) {

        final int aLength = a.numAttributes() - 1;
        final int bLength = b.numAttributes() - 1;

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
        jCosts[1] = Math.pow(b.value(0), 2);
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
            cost = Math.pow(b.value(j - 2) - b.value(j - 1), 2);
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
        iCost = Math.pow(a.value(0), 2);
        double min = Double.POSITIVE_INFINITY;
        if(start == 0) {
            row[0] = prevRow[0] + iCost;
            min = row[0];
            start++;
        }
        for(int j = start; j <= end; j++) {
            dist = Math.pow(a.value(0) - b.value(j - 1), 2);
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
            iCost = Math.pow(a.value(i - 2) - a.value(i - 1), 2);
            if(start == 0) {
                cost = prevRow[0] + iCost;
                row[0] = cost;
                min = Math.min(min, cost);
                start++;
            }
            if(start == 1) {
                dist = Math.pow(a.value(i - 1) - b.value(0), 2);
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
                dist = Math.pow(a.value(i - 1) - b.value(j - 1), 2) + Math.pow(a.value(i - 2) - b.value(j - 2), 2);
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
        ParamHandlerUtils.setParam(param, NU_FLAG, this::setNu, Double.class);
        ParamHandlerUtils.setParam(param, LAMBDA_FLAG, this::setLambda, Double.class);
    }

}
