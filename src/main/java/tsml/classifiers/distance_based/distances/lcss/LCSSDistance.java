package tsml.classifiers.distance_based.distances.lcss;

import tsml.classifiers.distance_based.distances.MatrixBasedDistanceMeasure;
import tsml.classifiers.distance_based.distances.dtw.DTW;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import tsml.data_containers.TimeSeries;
import tsml.data_containers.TimeSeriesInstance;

import java.util.Arrays;

/**
 * LCSS distance measure.
 * <p>
 * Contributors: goastler
 */
public class LCSSDistance extends MatrixBasedDistanceMeasure {
    
    // delta === warp
    // epsilon === diff between two values before they're considered the same AKA tolerance
    
    private double epsilon = 0.01;
    private double window = 1;

    public static final String EPSILON_FLAG = "e";
    public static final String WINDOW_FLAG = DTW.WINDOW_FLAG;

    public double getEpsilon() {
        return epsilon;
    }

    public void setEpsilon(double epsilon) {
        this.epsilon = epsilon;
    }

    private boolean approxEqual(TimeSeriesInstance a, int aIndex, TimeSeriesInstance b, int bIndex) {
        double sum = 0;
        for(int i = 0; i < a.getNumDimensions(); i++) {
            final TimeSeries aDim = a.get(i);
            final TimeSeries bDim = b.get(i);
            final Double aValue = aDim.get(aIndex);
            final Double bValue = bDim.get(bIndex);
            if(Math.abs(aValue - bValue) > epsilon) {
                return false;
            }
        }
        return true;
    }

    @Override
    public double distance(TimeSeriesInstance a, TimeSeriesInstance b, double limit) {
        
        // make a the longest time series
        if(a.getMaxLength() < b.getMaxLength()) {
            TimeSeriesInstance tmp = a;
            a = b;
            b = tmp;
        }
        
        final int aLength = a.getMaxLength();
        final int bLength = b.getMaxLength();
        setup(aLength, bLength, true);
        
        // 22/10/19 goastler - limit LCSS such that if any value in the current window is larger than the limit then we can stop here, no point in doing the extra work
        if(limit != Double.POSITIVE_INFINITY) { // check if there's a limit set
            // if so then reverse engineer the max LCSS distance and replace the limit
            // this is just the inverse of the return value integer rounded to an LCSS distance
            limit = (1 - limit) * Math.min(aLength, bLength);
            // is potentially slightly too low, causing *early* early abandon
        }
        
        // step is the increment of the mid point for each row
        final double step = (double) (bLength - 1) / (aLength - 1);
        final double windowSize = this.window * bLength;

        // row index
        int i = 0;

        // start and end of window
        int start = 0;
        double mid = 0;
        int end = Math.min(bLength - 1, (int) Math.floor(windowSize));
        int prevEnd; // store end of window from previous row to fill in shifted space with inf
        double[] row = getRow(i);
        double[] prevRow;

        // col index
        int j = start;
        // process top left sqaure of mat
        double min = row[j] = approxEqual(a, i, b, j) ? 1 : 0;
        j++;
        // compute the first row
        for(; j <= end; j++) {
            if(approxEqual(a, i, b, j)) {
                row[j] = 1;
            } else {
                row[j] = row[j - 1];
            }
            min = Math.min(min, row[j]);
        }
        if(min > limit) return Double.POSITIVE_INFINITY; // quit if beyond limit
        i++;
        
        // process remaining rows
        for(; i < aLength; i++) {
            // reset min for the row
            min = Double.POSITIVE_INFINITY;
            // change rows
            prevRow = row;
            row = getRow(i);

            // start, end and mid of window
            prevEnd = end;
            mid = i * step;
            // if using variable length time series and window size is fractional then the window may part cover an 
            // element. Any part covered element is truncated from the window. I.e. mid point of 5.5 with window of 2.3
            // would produce a start point of 2.2. The window would start from index 3 as it does not fully cover index
            // 2. The same thing happens at the end, 5.5 + 2.3 = 7.8, so the end index is 7 as it does not fully cover 8
            start = Math.max(0, (int) Math.ceil(mid - windowSize));
            end = Math.min(bLength - 1, (int) Math.floor(mid + windowSize));
            j = start;

            // set the values above the current row and outside of previous window to inf
            Arrays.fill(prevRow, prevEnd + 1, end + 1, Double.NEGATIVE_INFINITY);
            // set the value left of the window to inf
            if(j > 0) row[j - 1] = Double.NEGATIVE_INFINITY;
            
            // if assessing the left most column then only mapping option is top - not left or topleft
            if(j == 0) {
                if(approxEqual(a, i, b, j)) {
                    row[j] = 1;
                } else {
                    row[j] = prevRow[start];
                }
                min = Math.min(min, row[j++]);
            }
            
            // compute the distance for each cell in the row
            for(; j <= end; j++) {
                if(approxEqual(a, i, b, j)) {
                    row[j] = prevRow[j - 1] + 1;
                } else {
                    // note that the below is an edge case fix. LCSS algorithmically doesn't consider the topLeft cell
                    // when computing the max sequence. However, when a harsh enough window is applied the top and left
                    // cell become neg inf, causing issues as the max of neg inf and neg inf is neg inf. Therefore, we
                    // include topLeft in the max sequence candidates. In the case of a harsh window, this finds the
                    // value in the topLeft cell as max rather than neg inf in left or top. In non-harsh window cases,
                    // the topLeft value is always the same as left or topLeft (because they were not approx equal) or 
                    // -1 less (because they were approx equal, hence a +1 occurred. As it's always equal to or less, it
                    // has no effect upon the max operation under non-harsh window circumstances.
                    row[j] = Math.max(row[j - 1], Math.max(prevRow[j], prevRow[j - 1]));
                }
                min = Math.min(min, row[j]);
            }
            
            if(min > limit) return Double.POSITIVE_INFINITY; // quit if beyond limit
        }
        
        // last value in the current row is the distance
        final double distance = 1d - row[row.length - 1] / Math.min(aLength, bLength);
        teardown();
        return distance;
    }

    @Override protected double getFillerValue() {
        return Double.NEGATIVE_INFINITY; // LCSS maximises the subsequence count, so fill cost matrix with neg inf to begin with
    }

    @Override
    public ParamSet getParams() {
        return super.getParams().add(WINDOW_FLAG, window).add(EPSILON_FLAG, epsilon);
    }

    @Override
    public void setParams(final ParamSet param) throws Exception {
        super.setParams(param);
        setEpsilon(param.get(EPSILON_FLAG, getEpsilon()));
        setWindow(param.get(WINDOW_FLAG, getWindow()));
    }

    public double getWindow() {
        return window;
    }

    public void setWindow(final double window) {
        this.window = window;
    }
}
