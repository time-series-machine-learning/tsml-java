package tsml.classifiers.distance_based.distances.erp;

import tsml.classifiers.distance_based.distances.MatrixBasedDistanceMeasure;
import tsml.classifiers.distance_based.utils.collections.params.ParamHandlerUtils;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import tsml.data_containers.TimeSeries;
import tsml.data_containers.TimeSeriesInstance;
import utilities.Utilities;

import java.util.Arrays;

import static tsml.classifiers.distance_based.distances.dtw.DTW.WINDOW_SIZE_FLAG;
import static utilities.ArrayUtilities.*;

/**
 * ERP distance measure.
 * <p>
 * Contributors: goastler
 */
public class ERPDistance extends MatrixBasedDistanceMeasure {

    public static final String G_FLAG = "g";
    private double g = 0.01;
    private double windowSize = 1;
    
    public double getG() {
        return g;
    }

    public void setG(double g) {
        this.g = g;
    }
    
    public double cost(final TimeSeriesInstance a, final int aIndex) {
        double sum = 0;
        for(int i = 0; i < a.getNumDimensions(); i++) {
            final TimeSeries aDim = a.get(i);
            final Double aValue = aDim.get(aIndex);
            final double sqDiff = Math.pow(aValue - g, 2);
            sum += sqDiff;
        }
        return sum;
    }
    
    public double cost(TimeSeriesInstance a, int aIndex, TimeSeriesInstance b, int bIndex) {
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
    
    @Override
    public double distance(final TimeSeriesInstance a, final TimeSeriesInstance b, final double limit) {
        final int aLength = a.getMaxLength();
        final int bLength = b.getMaxLength();
        setup(aLength, bLength, true);
        final double lengthRatio = (double) bLength / aLength;
        final double windowSize = this.windowSize * bLength;
        
        // start and end of window
        int start = 0;
        int j = start;
        int i = 0;
        double mid;
        int end =  (int) Math.min(bLength - 1, Math.ceil(windowSize));
        int prevEnd;
        double[] row = getRow(0);
        double[] prevRow;
        
        // process top left sqaure of mat
        double min = row[j++] = 0; // top left cell is always zero
        // compute the first row
        for(; j <= end; j++) {
            row[j] = row[j - 1] + cost(b, j);
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
            Arrays.fill(prevRow, prevEnd + 1, end + 1, Double.POSITIVE_INFINITY);
            // set the value left of the window to inf
            if(j > 0) row[j - 1] = Double.POSITIVE_INFINITY;
            
            // if assessing the left most column then only mapping option is top - not left or topleft
            if(j == 0) {
                row[j] = prevRow[j] + cost(a, i);
                min = Math.min(min, row[j++]);
            }
            
            // compute the distance for each cell in the row
            for(; j <= end; j++) {
                final double topLeft = prevRow[j - 1] + cost(a, i, b, j);
                final double left = row[j - 1] + cost(b, j);
                final double top = prevRow[j] + cost(a, i);
                if(topLeft > left && left < top) {
                    // del
                    row[j] = left;
                } else if(topLeft > top && top < left) {
                    // ins
                    row[j] = top;
                } else {
                    // match
                    row[j] = topLeft;
                }
                min = Math.min(min, row[j]);
            }
            if(min > limit) return Double.POSITIVE_INFINITY; // quit if beyond limit
        }
        // last value in the current row is the distance
        final double distance = row[bLength - 1];
        teardown();
        return distance;
    }

    @Override
    public ParamSet getParams() {
        return super.getParams().add(WINDOW_SIZE_FLAG, windowSize).add(G_FLAG, g);
    }

    @Override
    public void setParams(final ParamSet param) throws Exception {
        super.setParams(param);
        ParamHandlerUtils.setParam(param, G_FLAG, this::setG, Double::parseDouble);
        ParamHandlerUtils.setParam(param, WINDOW_SIZE_FLAG, this::setWindowSize, Double::parseDouble);
    }

    public double getWindowSize() {
        return windowSize;
    }

    public void setWindowSize(final double windowSize) {
        this.windowSize = Utilities.requirePercentage(windowSize);
    }
}
