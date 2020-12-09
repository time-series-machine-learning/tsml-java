package tsml.classifiers.distance_based.distances.dtw;

import tsml.classifiers.distance_based.distances.BaseDistanceMeasure;
import tsml.classifiers.distance_based.utils.collections.params.ParamHandlerUtils;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import tsml.classifiers.distance_based.utils.strings.StrUtils;
import tsml.data_containers.TimeSeriesInstance;
import utilities.Utilities;

import java.util.Arrays;
import java.util.stream.DoubleStream;

import static utilities.ArrayUtilities.*;

/**
 * DTW distance measure.
 * <p>
 * Contributors: goastler
 */
public class DTWDistance extends BaseDistanceMeasure implements DTW {
    public static String WINDOW_SIZE_FLAG = "w";
    
    public static double cost(final TimeSeriesInstance a, final int aIndex, final TimeSeriesInstance b, final int bIndex) {
        final double[] aSlice = a.getVSliceArray(aIndex);
        final double[] bSlice = b.getVSliceArray(bIndex);
        subtract(aSlice, bSlice);
        pow(aSlice, 2);
        return sum(aSlice);
    }
    
    private double windowSize = 1;

    @Override public void setWindowSize(final double windowSize) {
        this.windowSize = Utilities.requirePercentage(windowSize);
    }

//    @Override public double getWindowSize() {
//        return windowSize;
//    }


    @Override public WindowParameter getWindowParameter() {
        return null;
    }

    public double distance(TimeSeriesInstance a, TimeSeriesInstance b, final double limit) {
        // check data is in expected format
        checkData(a, b, limit);

        final int aLength = a.getMaxLength();
        final int bLength = b.getMaxLength();
        final double lengthRatio = (double) bLength / aLength;
        final double windowSize = this.windowSize * bLength;
        
        final boolean generateDistanceMatrix = isGenerateDistanceMatrix();
        final double[][] matrix = generateDistanceMatrix ? new double[aLength][bLength] : null;
        setDistanceMatrix(matrix);
        
        // row by row matrix traversal. Use len + 1 as left most column should be pos inf, same with first row. Pos inf's are used to stop the mapping over the top/left edge of the matrix when considering top/left/topleft
        /* i.e. the distance matrix may end up being
            2,3,4,5
            2,4,6,7
            3,7,5,8
            4,5,6,6
            
           but we're padding with a top row and left column of pos inf
           
            0,inf,inf,inf,inf
            inf,2,3,4,5
            inf,2,4,6,7
            inf,3,7,5,8
            inf,4,5,6,6
            
           this means the left and top row can consider left / top / topleft neighbours without having to check for index out of bounds.
           the [0][0]'th element is set to zero to allow the [1][1]'th to warp.
           note the distance matrix is NOT populated with the padded top row and left col of infs as it's just a implementation detail.
        
         */
        double[] row = DoubleStream.generate(() -> Double.POSITIVE_INFINITY).limit(bLength + 1).toArray();
        row[0] = 0;
        // make the prevRow pos inf to deny mapping
        double[] prevRow = DoubleStream.generate(() -> Double.POSITIVE_INFINITY).limit(bLength + 1).toArray();
        // track the previous window's start and end points. Used to fill in the gaps between subsequent reuses of rows
        int prevStart = 1;
        int prevPrevStart = 1;
        // for each row
        for(int i = 0; i < aLength; i++) {
            // make a new row (just reuse the previous)
            {
                double[] tmp = prevRow;
                prevRow = row;
                row = tmp;
            }
            // the min for this row begins with pos inf (i.e. no min seen yet)
            double min = Double.POSITIVE_INFINITY;
            // find the start and end of the window for this row
            final double midPoint = i * lengthRatio;
            // shift both start and end left by 1 to account for left-most pos inf
            final int start = (int) Math.max(0, Math.ceil(midPoint - windowSize)) + 1;
            final int end = (int) Math.min(bLength - 1, Math.floor(midPoint + windowSize)) + 1;
            // set the left-most value to pos inf
            row[start - 1] = Double.POSITIVE_INFINITY;
            // for each column
            for(int j = start; j <= end; j++) {
                // compute mapping - all accesses shifted left by one to account for left most column of pos infs
                final double topLeft = prevRow[j - 1];
                final double left = row[j - 1];
                final double top = prevRow[j];
                final double cellCost = cost(a, i, b, j - 1); // offset by 1 because of left pos inf
                final double cost = Math.min(top, Math.min(left, topLeft)) + cellCost;
                row[j] = cost;
                min = Math.min(min, cost);
            }
            // save row to matrix
            if(generateDistanceMatrix) {
                // fill in the gap between prevStart - start
                Arrays.fill(row, prevPrevStart, start, Double.POSITIVE_INFINITY);
                // also fill in the cell on the prev row. This gets missed due to every-other-row usage pattern / reuse of rows
//                Arrays.fill(row, prevStart, start, Double.POSITIVE_INFINITY);
//                prevRow[start - 1] = Double.POSITIVE_INFINITY;
                // populate current row
                System.arraycopy(row, 1, matrix[i], 0, bLength);
                // update start point history
                prevPrevStart = prevStart;
                prevStart = start;
            }
            // check if limit exceeded
            if(min > limit) {
                return Double.POSITIVE_INFINITY;
            }
        }
        // last value in the current row is the distance
        return row[row.length - 1];
    }

    @Override public ParamSet getParams() {
        return new ParamSet().add(WINDOW_SIZE_FLAG, windowSize);
    }

    @Override public void setParams(final ParamSet paramSet) throws Exception {
//        ParamHandlerUtils.setParam(paramSet, WINDOW_SIZE_FLAG, this::setWindowSize);
    }

}
