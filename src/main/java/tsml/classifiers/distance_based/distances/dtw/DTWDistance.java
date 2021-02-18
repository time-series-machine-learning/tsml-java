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
            final Double aValue = aDim.get(aIndex);
            final Double bValue = bDim.get(bIndex);
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
        // pad an extra inf row and inf col
        setup(aLength + 1, bLength + 1, true);
        // step is the increment of the mid point for each row
        final double step = (double) (bLength - 1) / (aLength - 1);
        final double windowSize = this.window * bLength;

        // start and end of window
        int start, prevEnd, end = 0;
        double mid, min;
        double[] prevRow;
        double[] row = getRow(0);
        
        // anchor point of zero in the padding inf col / row
        row[0] = 0;

        // process remaining rows
        for(int i = 1; i < aLength + 1; i++) {
            // reset min for the row
            min = Double.POSITIVE_INFINITY;

            // start, end and mid of window
            prevEnd = end;
            mid = (i - 1) * step + 1;
            start = Math.max(1, (int) Math.ceil(mid - windowSize));
            end = Math.min(bLength, (int) Math.floor(mid + windowSize));
            
            // change rows
            prevRow = row;
            row = getRow(i);
            row[start - 1] = Double.POSITIVE_INFINITY;

            // set the top values outside of window to inf
            Arrays.fill(prevRow, prevEnd + 1, end + 1, Double.POSITIVE_INFINITY);

            // compute the distance for each cell in the row
            for(int j = start; j <= end; j++) {
                row[j] = Math.min(prevRow[j], Math.min(row[j - 1], prevRow[j - 1])) + cost(a, i - 1, b, j - 1);
                min = Math.min(min, row[j]);
            }

            // quit if beyond limit
            if(min > limit) return Double.POSITIVE_INFINITY;
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
