package tsml.classifiers.distance_based.distances.erp;

import tsml.classifiers.distance_based.distances.DoubleMatrixBasedDistanceMeasure;
import tsml.classifiers.distance_based.distances.WarpingDistanceMeasure;
import tsml.classifiers.distance_based.distances.WarpingParameter;
import tsml.classifiers.distance_based.utils.collections.params.ParamHandlerUtils;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import weka.core.Instance;

/**
 * ERP distance measure.
 * <p>
 * Contributors: goastler
 */
public class ERPDistance extends DoubleMatrixBasedDistanceMeasure implements WarpingDistanceMeasure {

    public static final String G_FLAG = "g";
    public static final String WINDOW_SIZE_FLAG = WarpingParameter.WINDOW_SIZE_FLAG;
    public static final String WINDOW_SIZE_PERCENTAGE_FLAG = WarpingParameter.WINDOW_SIZE_PERCENTAGE_FLAG;
    private double g = 0;
    private final WarpingParameter warpingParameter = new WarpingParameter();

    public double getG() {
        return g;
    }

    public void setG(double g) {
        this.g = g;
    }

    @Override
    public double findDistance(final Instance a, final Instance b, final double limit) {

        int aLength = a.numAttributes() - 1;
        int bLength = b.numAttributes() - 1;

        final boolean generateDistanceMatrix = isGenerateDistanceMatrix();
        final double[][] matrix = generateDistanceMatrix ? new double[aLength][bLength] : null;
        setDistanceMatrix(matrix);

        // Current and previous columns of the matrix
        double[] row = new double[bLength];
        double[] prevRow = new double[bLength];
        double min;
        // size of edit distance band
        // bandsize is the maximum allowed distance to the diagonal
        final int windowSize = findWindowSize(aLength);

        int start = 1;
        int end = Math.min(bLength - 1, windowSize);
        if(end + 1 < bLength) {
            row[end + 1] = Double.POSITIVE_INFINITY;
        }
        row[0] = 0; // top left cell of matrix is always 0
        // populate first row
        for(int j = start; j <= end; j++) {
            final double cost = row[j - 1] + Math.pow(b.value(j) - g, 2);
            row[j] = cost;
            // no need to update min as top left cell is already zero, can't get lower
        }
        // populate matrix
        if(generateDistanceMatrix) {
            System.arraycopy(row, 0, matrix[0], 0, row.length);
        }
        // no need to check for early abandon here as the min is zero because of the top left cell
        // populate remaining rows
        for(int i = 1; i < aLength; i++) {
            // Swap current and prevRow arrays. We'll just overwrite the new row.
            {
                double[] temp = prevRow;
                prevRow = row;
                row = temp;
            }
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
            // when l == 0 neither left nor top left can be picked, therefore it must use top
            if(start == 0) {
                final double cost = prevRow[start] + Math.pow(a.value(i) - g, 2);
                row[start] = cost;
                min = Math.min(min, cost);
                start++;
            }
            for(int j = start; j <= end; j++) {
                // compute squared distance of feature vectors
                final double v1 = a.value(i);
                final double v2 = b.value(j);
                final double leftPenalty = Math.pow(v1 - g, 2);
                final double topPenalty = Math.pow(v2 - g, 2);
                final double topLeftPenalty = Math.pow(v1 - v2, 2);
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
            if(generateDistanceMatrix) {
                System.arraycopy(row, 0, matrix[i], 0, row.length);
            }
            if(min > limit) {
                return Double.POSITIVE_INFINITY;
            }
        }

        return row[bLength - 1];
    }

    @Override
    public ParamSet getParams() {
        return super.getParams().addAll(warpingParameter.getParams()).add(G_FLAG, g);
    }

    @Override
    public void setParams(final ParamSet param) throws Exception {
        super.setParams(param);
        warpingParameter.setParams(param);
        ParamHandlerUtils.setParam(param, G_FLAG, this::setG, Double.class);
    }

    @Override public int findWindowSize(final int aLength) {
        return warpingParameter.findWindowSize(aLength);
    }

    @Override public int getWindowSize() {
        return warpingParameter.getWindowSize();
    }

    @Override public void setWindowSize(final int windowSize) {
        warpingParameter.setWindowSize(windowSize);
    }

    @Override public double getWindowSizePercentage() {
        return warpingParameter.getWindowSizePercentage();
    }

    @Override public void setWindowSizePercentage(final double windowSizePercentage) {
        warpingParameter.setWindowSizePercentage(windowSizePercentage);
    }

    @Override public boolean isWindowSizeInPercentage() {
        return warpingParameter.isWindowSizeInPercentage();
    }
}
