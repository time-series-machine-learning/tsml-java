package tsml.classifiers.distance_based.distances.msm;

import tsml.classifiers.distance_based.distances.MatrixBasedDistanceMeasure;
import tsml.classifiers.distance_based.utils.collections.params.ParamHandlerUtils;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import tsml.data_containers.TimeSeries;
import tsml.data_containers.TimeSeriesInstance;

import java.util.Arrays;
import java.util.stream.IntStream;

import static utilities.ArrayUtilities.*;

/**
 * MSM distance measure.
 * <p>
 * Contributors: goastler
 */
public class MSMDistance extends MatrixBasedDistanceMeasure {
    
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

    /**
     * Find the cost for doing a move / split / merge for the univariate case.
     * @param newPoint
     * @param x
     * @param y
     * @return
     */
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

    /**
     * Find the cost for doing a move / split / merge on the multivariate case.
     * @param a
     * @param aIndex
     * @param b
     * @param bIndex
     * @param c
     * @param cIndex
     * @return
     */
    private double cost(final TimeSeriesInstance a, final int aIndex, final TimeSeriesInstance b, final int bIndex, final TimeSeriesInstance c, final int cIndex) {
        double sum = 0;
        for(int i = 0; i < a.getNumDimensions(); i++) {
            final TimeSeries aDim = a.get(i);
            final TimeSeries bDim = b.get(i);
            final TimeSeries cDim = c.get(i);
            final Double aValue = aDim.get(aIndex);
            final Double bValue = bDim.get(bIndex);
            final Double cValue = cDim.get(cIndex);
            sum += findCost(aValue, bValue, cValue);
        }
        return sum;
    }

    /**
     * Find the cost for a individual cell. This is used when there's no alignment change and values are mapping directly to another.
     * @param a
     * @param aIndex
     * @param b
     * @param bIndex
     * @return
     */
    private double directCost(final TimeSeriesInstance a, final int aIndex, final TimeSeriesInstance b, final int bIndex) {
        double sum = 0;
        for(int i = 0; i < a.getNumDimensions(); i++) {
            final TimeSeries aDim = a.get(i);
            final TimeSeries bDim = b.get(i);
            final Double aValue = aDim.get(aIndex);
            final Double bValue = bDim.get(bIndex);
            sum += Math.abs(aValue - bValue);
        }
        return sum;
    }

    @Override
    public double distance(TimeSeriesInstance a, TimeSeriesInstance b, final double limit) {
        final int aLength = a.getMaxLength();
        final int bLength = b.getMaxLength();
        setup(aLength, bLength, true);
        final double lengthRatio = (double) bLength / aLength;
        final double windowSize = 1d * bLength;
        
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
        double min = row[j] = directCost(a, i, b, j);
        j++;
        // compute the first row
        for(; j <= end; j++) {
            row[j] = row[j - 1] + cost(b, j, a, i, b, j - 1);
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
                row[j] = prevRow[j] + cost(a, i, a, i - 1, b, j);
                min = Math.min(min, row[j++]);
            }
            
            // compute the distance for each cell in the row
            for(; j <= end; j++) {
                final double topLeft = prevRow[j - 1] + directCost(a, i, b, j);
                final double top = prevRow[j] + cost(a, i, a, i - 1, b, j);
                final double left = row[j - 1] + cost(b, j, a, i, b, j - 1);
                row[j] = Math.min(top, Math.min(left, topLeft));
                min = Math.min(min, row[j]);
            }
            if(min > limit) return Double.POSITIVE_INFINITY; // quit if beyond limit
        }
        // last value in the current row is the distance
        final double distance = row[row.length - 1];
        teardown();
        return distance;
    }

    @Override
    public ParamSet getParams() {
        return super.getParams().add(C_FLAG, c);
    }

    @Override
    public void setParams(final ParamSet param) throws Exception {
        super.setParams(param);
        ParamHandlerUtils.setParam(param, C_FLAG, this::setC, Double::parseDouble);
    }
}
