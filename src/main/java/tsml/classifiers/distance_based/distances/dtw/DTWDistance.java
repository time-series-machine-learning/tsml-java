package tsml.classifiers.distance_based.distances.dtw;

import tsml.classifiers.distance_based.distances.MatrixBasedDistanceMeasure;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import tsml.data_containers.TimeSeries;
import tsml.data_containers.TimeSeriesInstance;
import utilities.Utilities;

import java.util.Arrays;

/**
 * DTW distance measure.
 * <p>
 * Contributors: goastler
 */
public class DTWDistance extends MatrixBasedDistanceMeasure implements DTW {

    public static double cost(final TimeSeriesInstance a, final int aIndex, final TimeSeriesInstance b, final int bIndex) {
        double sum = 0;
        for(int i = 0; i < a.getNumDimensions(); i++) {
            final TimeSeries aDim = a.get(i);
            final TimeSeries bDim = b.get(i);
            final double aValue = aDim.get(aIndex);
            final double bValue = bDim.get(bIndex);
            final double sqDiff = Math.pow(aValue - bValue, 2);
            sum += sqDiff;
        }
        return sum;
    }

    private double window = 1;

    @Override public void setWindow(final double window) {
        this.window = Utilities.requirePercentage(window);
    }

    @Override public double getWindow() {
        return window;
    }

    public double distance(TimeSeriesInstance a, TimeSeriesInstance b, final double limit) {

        // make a the longest time series
        if(a.getMaxLength() < b.getMaxLength()) {
            TimeSeriesInstance tmp = a;
            a = b;
            b = tmp;
        }
        
        final int aLength = a.getMaxLength();
        final int bLength = b.getMaxLength();
        setup(aLength, bLength, true);
        
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
        // process the first row (can only warp left - not top/topLeft)
        double min = row[j++] = cost(a, 0, b, 0); // process top left sqaure of mat
        // compute the first row
        for(; j <= end; j++) {
            row[j] = row[j - 1] + cost(a, i, b, j);
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
            Arrays.fill(prevRow, prevEnd + 1, end + 1, Double.POSITIVE_INFINITY);
            // set the value left of the window to inf
            if(j > 0) row[j - 1] = Double.POSITIVE_INFINITY;

            // if assessing the left most column then only mapping option is top - not left or topleft
            if(j == 0) {
                row[j] = prevRow[j] + cost(a, i, b, j);
                min = Math.min(min, row[j++]);
            }

            // compute the distance for each cell in the row
            for(; j <= end; j++) {
                row[j] = Math.min(prevRow[j], Math.min(row[j - 1], prevRow[j - 1])) + cost(a, i, b, j);
                min = Math.min(min, row[j]);
            }
            
            if(min > limit) return Double.POSITIVE_INFINITY; // quit if beyond limit
        }

        // last value in the current row is the distance
        final double distance = row[row.length - 1];
        teardown();
        return distance;
    }

    @Override public ParamSet getParams() {
        return new ParamSet().add(WINDOW_FLAG, window);
    }

    @Override public void setParams(final ParamSet paramSet) throws Exception {
        setWindow(paramSet.get(WINDOW_FLAG, window));
    }

    public static void main(String[] args) {
        final DTWDistance dm = new DTWDistance();
        dm.setWindow(0.2);
        dm.setRecordCostMatrix(true);
        final double a = dm.distanceUnivariate(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, new double[]{1, 2, 3});
        System.out.println();
        System.out.println("-----");
        System.out.println();
        final double b = dm.distanceUnivariate(new double[]{1, 2, 3}, new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
        System.out.println(a);
        System.out.println(b);
        if(a != b) {
            System.out.println("not eq");
            System.out.println();
        }
    }
}
