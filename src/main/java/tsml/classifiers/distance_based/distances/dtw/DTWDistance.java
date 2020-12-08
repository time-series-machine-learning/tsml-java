package tsml.classifiers.distance_based.distances.dtw;


import org.apache.commons.lang3.ArrayUtils;
import tsml.classifiers.distance_based.distances.BaseDistanceMeasure;
import tsml.data_containers.TimeSeriesInstance;
import utilities.ArrayUtilities;

import java.util.stream.DoubleStream;

import static utilities.ArrayUtilities.*;

/**
 * DTW distance measure.
 * <p>
 * Contributors: goastler
 */
public class DTWDistance extends BaseDistanceMeasure implements DTW {

    private final WindowParameter windowParameter = new WindowParameter();

    @Override public WindowParameter getWindowParameter() {
        return windowParameter;
    }
    
    public static double cost(final TimeSeriesInstance a, final int aIndex, final TimeSeriesInstance b, final int bIndex) {
        final double[] aSlice = a.getVSliceArray(aIndex);
        final double[] bSlice = b.getVSliceArray(bIndex);
        subtract(aSlice, bSlice);
        pow(aSlice, 2);
        return sum(aSlice);
    }

    public double distance(final TimeSeriesInstance a, final TimeSeriesInstance b, final double limit) {
        
        runChecks(a, b, limit);
        
        final int aLength = a.getMaxLength();
        final int bLength = b.getMaxLength();
        
        final boolean generateDistanceMatrix = isGenerateDistanceMatrix();
        final double[][] matrix = generateDistanceMatrix ? new double[aLength][bLength] : null;
        setDistanceMatrix(matrix);

        // window should be somewhere from 0..len-1. window of 0 is ED, len-1 is Full DTW. Anything above is just
        // Full DTW
        final int windowSize = findWindowSize(aLength);

        // row by row matrix traversal. Use len + 1 as left most column should be pos inf, same with first row. Pos inf's are used to stop the mapping over the top/left edge of the matrix when considering top/left/topleft
        double[] row = DoubleStream.generate(() -> Double.POSITIVE_INFINITY).limit(bLength + 1).toArray();
        row[0] = 0;
        // make the prevRow pos inf to deny mapping
        double[] prevRow = DoubleStream.generate(() -> Double.POSITIVE_INFINITY).limit(bLength + 1).toArray();
        // init min to pos inf, i.e. no min seen yet
        double min;
        // for each row
        for(int i = 0; i < aLength; i++) {
            // make a new row (just reuse the previous)
            {
                double[] tmp = prevRow;
                prevRow = row;
                row = tmp;
            }
            // the min for this row begins with pos inf (i.e. no min seen yet)
            min = Double.POSITIVE_INFINITY;
            // find the start and end of the window for this row
            int start = Math.max(0, i - windowSize);
            int end = Math.min(bLength - 1, i + windowSize);
            // the left-most value is always pos inf to avoid mapping over the left hand edge of window
            row[start] = Double.POSITIVE_INFINITY;
            // for each column
            for(int j = start; j <= end; j++) {
                // compute mapping - all accesses shifted left by one to account for left most column of pos infs
                final double topLeft = prevRow[j];
                final double left = row[j];
                final double top = prevRow[j + 1];
                final double cost = Math.min(top, Math.min(left, topLeft)) + cost(a, i, b, j);
                row[j + 1] = cost;
                min = Math.min(min, cost);
            }
            prevRow[start] = Double.POSITIVE_INFINITY;
            // save row to matrix
            if(generateDistanceMatrix) {
                System.arraycopy(row, 1, matrix[i], 0, bLength);
            }
            // check if limit exceeded
            if(min > limit) {
                return Double.POSITIVE_INFINITY;
            }
        }
        // last value in the current row is the distance
        return row[row.length - 1];
        
//        // top left cell of matrix will simply be the sq diff
//        // min can be init'd to the top left cell
//        double min = cost(a, 0, b, 0);
//        row[0] = min;
//        // start and end of window
//        // start at the next cell of the first row
//        int start = 1;
//        // end at window or bLength, whichever smallest
//        int end = Math.min(bLength - 1, windowSize);
//        // must set the value before and after the window to inf if available as the following row will use these
//        // in top / left / top-left comparisons
//        if(end + 1 < bLength) {
//            row[end + 1] = Double.POSITIVE_INFINITY;
//        }
//        // the first row is populated from the sq diff + the cell before
//        for(int j = start; j <= end; j++) {
//            double cost = row[j - 1] + cost(a, 0, b, j);
//            row[j] = cost;
//            min = Math.min(min, cost);
//        }
//        if(generateDistanceMatrix) {
//            System.arraycopy(row, 0, matrix[0], 0, row.length);
//        }
//        // early abandon if work has been done populating the first row for >1 entry
//        if(min > limit) {
//            return Double.POSITIVE_INFINITY;
//        }
//        for(int i = 1; i < aLength; i++) {
//            // Swap current and prevRow arrays. We'll just overwrite the new row.
//            {
//                double[] temp = prevRow;
//                prevRow = row;
//                row = temp;
//            }
//            // reset the insideLimit var each row. if all values for a row are above the limit then early abandon
//            min = Double.POSITIVE_INFINITY;
//            // start and end of window
//            start = Math.max(0, i - windowSize);
//            end = Math.min(bLength - 1, i + windowSize);
//            // must set the value before and after the window to inf if available as the following row will use these
//            // in top / left / top-left comparisons
//            if(start - 1 >= 0) {
//                row[start - 1] = Double.POSITIVE_INFINITY;
//            }
//            if(end + 1 < bLength) {
//                row[end + 1] = Double.POSITIVE_INFINITY;
//            }
//            // if assessing the left most column then only top is the option - not left or left-top
//            if(start == 0) {
//                final double cost = prevRow[start] + cost(a, i, b, 0);
//                row[start] = cost;
//                min = Math.min(min, cost);
//                // shift to next cell
//                start++;
//            }
//            for(int j = start; j <= end; j++) {
//                // compute squared distance of feature vectors
//                final double topLeft = prevRow[j - 1];
//                final double left = row[j - 1];
//                final double top = prevRow[j];
//                final double cost = Math.min(top, Math.min(left, topLeft)) + cost(a, i, b, j);
//                row[j] = cost;
//                min = Math.min(min, cost);
//            }
//            if(generateDistanceMatrix) {
//                System.arraycopy(row, 0, matrix[i], 0, row.length);
//            }
//            if(min > limit) {
//                return Double.POSITIVE_INFINITY;
//            }
//        }
//        //Find the minimum distance at the end points, within the warping window.
//        return row[bLength - 1];
    }

}
