package tsml.classifiers.distance_based.distances.msm;

import tsml.classifiers.distance_based.distances.DoubleMatrixBasedDistanceMeasure;
import tsml.classifiers.distance_based.utils.collections.params.ParamHandlerUtils;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import weka.core.Instance;

/**
 * MSM distance measure.
 * <p>
 * Contributors: goastler
 */
public class MSMDistance
    extends DoubleMatrixBasedDistanceMeasure {


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

    @Override
    public double findDistance(Instance a, Instance b, final double limit) {

        int aLength = a.numAttributes() - 1;
        int bLength = b.numAttributes() - 1;

        final boolean generateDistanceMatrix = isGenerateDistanceMatrix();
        final double[][] matrix = generateDistanceMatrix ? new double[aLength][bLength] : null;
        setDistanceMatrix(matrix);

        final int windowSize = aLength;

        double[] row = new double[bLength];
        double[] prevRow = new double[bLength];
        // top left cell of matrix will simply be the sq diff
        double min = Math.abs(a.value(0) - b.value(0));
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
            double cost = row[j - 1] + findCost(b.value(j), a.value(0), b.value(j - 1));
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
            final double ai = a.value(i);
            final double aiPrev = a.value(i - 1);
            if(start == 0) {
                final double cost = prevRow[start] + findCost(ai, aiPrev, b.value(start));
                row[start] = cost;
                min = Math.min(min, cost);
                // shift to next cell
                start++;
            }
            for(int j = start; j <= end; j++) {
                final double bj = b.value(j);
                // compute squared distance of feature vectors
                final double topLeft = prevRow[j - 1] + Math.abs(ai - bj);
                final double top = prevRow[j] + findCost(ai, aiPrev, bj);
                final double left = row[j - 1] + findCost(bj, ai, b.value(j - 1));
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
        ParamHandlerUtils.setParam(param, C_FLAG, this::setC, Double.class);
    }
}
