package tsml.classifiers.distance_based.distances.wdtw;

import tsml.classifiers.distance_based.distances.MatrixBasedDistanceMeasure;
import tsml.classifiers.distance_based.utils.collections.params.ParamHandlerUtils;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import tsml.data_containers.TimeSeriesInstance;

import java.util.Arrays;

import static tsml.classifiers.distance_based.distances.dtw.DTWDistance.cost;

/**
 * WDTW distance measure.
 * <p>
 * Contributors: goastler
 */
public class WDTWDistance
    extends MatrixBasedDistanceMeasure implements WDTW {

    private double g = 0.05;
    private double[] weights = new double[0];

    @Override
    public double getG() {
        return g;
    }

    @Override
    public void setG(double g) {
        if(g != this.g) {
            // reset the weights if g changes
            weights = new double[0];
        }
        this.g = g;
    }
    
    private void generateWeights(int length) {
        if(weights.length < length) {
            final double halfLength = (double) length / 2;
            double[] oldWeights = weights;
            weights = new double[length];
            System.arraycopy(oldWeights, 0, weights, 0, oldWeights.length);
            for(int i = oldWeights.length; i < length; i++) {
                weights[i] = 1d / (1d + Math.exp(-g * (i - halfLength)));
            }
        }
    }

    @Override
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
        final double window = 1;
        final double windowSize = window * bLength;

        // generate weights for soft weighting of costs
        generateWeights(Math.max(aLength, bLength));

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
        
        // process top left cell of mat
        double min = row[j] = weights[j] * cost(a, i, b, j);
        j++;
        // compute the first row
        for(; j <= end; j++) {
            row[j] = row[j - 1] + weights[j] * cost(a, i, b, j);
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
                row[j] = prevRow[j] + weights[Math.abs(i - j)] * cost(a, i, b, j);
                min = Math.min(min, row[j++]);
            }
            // compute the distance for each cell in the row
            for(; j <= end; j++) {
                row[j] = Math.min(prevRow[j], Math.min(row[j - 1], prevRow[j - 1])) + weights[Math.abs(i - j)] * cost(a, i, b, j);;
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
        return super.getParams().add(WDTW.G_FLAG, g);
    }

    @Override
    public void setParams(final ParamSet param) throws Exception {
        super.setParams(param);
        setG(param.get(G_FLAG, getG()));
    }

}
