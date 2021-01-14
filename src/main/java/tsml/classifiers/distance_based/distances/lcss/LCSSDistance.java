package tsml.classifiers.distance_based.distances.lcss;

import tsml.classifiers.distance_based.distances.MatrixBasedDistanceMeasure;
import tsml.classifiers.distance_based.utils.collections.params.ParamHandlerUtils;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import tsml.data_containers.TimeSeries;
import tsml.data_containers.TimeSeriesInstance;

import java.util.Arrays;

import static tsml.classifiers.distance_based.distances.dtw.DTW.WINDOW_SIZE_FLAG;
import static utilities.ArrayUtilities.*;

/**
 * LCSS distance measure.
 * <p>
 * Contributors: goastler
 */
public class LCSSDistance extends MatrixBasedDistanceMeasure {
    
    // delta === warp
    // epsilon === diff between two values before they're considered the same AKA tolerance
    
    private double epsilon = 0.01;
    private double windowSize = 1;

    public static final String EPSILON_FLAG = "e";

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
    public double distance(final TimeSeriesInstance a, final TimeSeriesInstance b, double limit) {
        final int aLength = a.getMaxLength();
        final int bLength = b.getMaxLength();
        setup(aLength, bLength, true);
        final double lengthRatio = (double) bLength / aLength;
        final double windowSize = this.windowSize * bLength;
        
        // 22/10/19 goastler - limit LCSS such that if any value in the current window is larger than the limit then we can stop here, no point in doing the extra work
        if(limit != Double.POSITIVE_INFINITY) { // check if there's a limit set
            // if so then reverse engineer the max LCSS distance and replace the limit
            // this is just the inverse of the return value integer rounded to an LCSS distance
            limit = (1 - limit) * Math.min(aLength, bLength);
            // is potentially slightly too low, causing *early* early abandon
        }
        
        // start and end of window
        int start = 0;
        int j = start;
        int i = 0;
        double mid;
        int end =  (int) Math.min(bLength - 1, Math.ceil(windowSize));
        int prevEnd;
        double[] row = getRow(i);
        double[] prevRow;
        
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
            
            // start, end and mid of window
            prevEnd = end;
            mid = i * lengthRatio;
            start = (int) Math.max(0, Math.floor(mid - windowSize));
            j = start;
            end = (int) Math.min(bLength - 1, Math.ceil(mid + windowSize));
            
            // change rows
            prevRow = row;
            row = getRow(i);
            
            // set the top values outside of window to inf
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
                    row[j] = Math.max(row[j - 1], Math.max(prevRow[j], prevRow[j - 1]));
                }
                min = Math.min(min, row[j]);
            }
            if(min > limit) return Double.POSITIVE_INFINITY; // quit if beyond limit
        }
        // last value in the current row is the distance
        return 1d - row[row.length - 1] / Math.min(aLength, bLength);
    }

    @Override
    public ParamSet getParams() {
        return super.getParams().add(WINDOW_SIZE_FLAG, windowSize).add(EPSILON_FLAG, epsilon);
    }

    @Override
    public void setParams(final ParamSet param) throws Exception {
        ParamHandlerUtils.setParam(param, EPSILON_FLAG, this::setEpsilon, Double::parseDouble);
        ParamHandlerUtils.setParam(param, WINDOW_SIZE_FLAG, this::setWindowSize, Double::parseDouble);
        super.setParams(param);
    }

    public double getWindowSize() {
        return windowSize;
    }

    public void setWindowSize(final double windowSize) {
        this.windowSize = windowSize;
    }
}
