package tsml.classifiers.distance_based.distances.dtw;

import tsml.classifiers.distance_based.distances.MatrixBasedDistanceMeasure;
import tsml.classifiers.distance_based.distances.transformed.BaseTransformDistanceMeasure;
import tsml.classifiers.distance_based.distances.transformed.TransformDistanceMeasure;
import tsml.classifiers.distance_based.utils.collections.params.ParamHandlerUtils;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import tsml.data_containers.TimeSeries;
import tsml.data_containers.TimeSeriesInstance;
import tsml.transformers.Derivative;
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
    
    private double windowSize = 1;

    @Override public void setWindowSize(final double windowSize) {
        this.windowSize = Utilities.requirePercentage(windowSize);
    }

    @Override public double getWindowSize() {
        return windowSize;
    }

    public double distance(TimeSeriesInstance a, TimeSeriesInstance b, final double limit) {

        final int aLength = a.getMaxLength();
        final int bLength = b.getMaxLength();
        setup(aLength, bLength, true);
        final double lengthRatio = (double) bLength / aLength;
        final double windowSize = this.windowSize * bLength;
        
        // start and end of window
        int start = 0;
        int j = start;
        double mid;
        int end =  (int) Math.min(bLength - 1, Math.ceil(windowSize));
        int prevEnd;
        int i = 0;
        double[] row = getRow(i);
        double[] prevRow;
        
        // process top left sqaure of mat
        double min = row[j++] = cost(a, 0, b, 0);
        // compute the first row
        for(; j <= end; j++) {
            row[j] = row[j - 1] + cost(a, 0, b, j);
            min = Math.min(min, row[j]);
        }
        if(min > limit) return Double.POSITIVE_INFINITY; // quit if beyond limit
        i++;
        
        // process remaining rows
        for(; i < aLength; i++) {
            // reset min for the row
            min = Double.POSITIVE_INFINITY;
            
            // start, end and mid of window
            prevEnd = end;
            mid = i * lengthRatio;
            start = (int) Math.max(0, Math.floor(mid - windowSize));
            end = (int) Math.min(bLength - 1, Math.ceil(mid + windowSize));
            j = start;
            
            // change rows
            prevRow = row;
            row = getRow(i);
            
            // set the top values outside of window to inf
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
            
            // quit if beyond limit
            if(min > limit) return Double.POSITIVE_INFINITY;
        }
        
        // last value in the current row is the distance
        final double distance = row[row.length - 1];
        teardown();
        return distance;
    }

    @Override public ParamSet getParams() {
        return new ParamSet().add(WINDOW_SIZE_FLAG, windowSize);
    }

    @Override public void setParams(final ParamSet paramSet) throws Exception {
        ParamHandlerUtils.setParam(paramSet, WINDOW_SIZE_FLAG, this::setWindowSize, Double::parseDouble);
    }

}
