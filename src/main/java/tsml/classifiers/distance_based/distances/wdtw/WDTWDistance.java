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
    public double distance(final TimeSeriesInstance a, final TimeSeriesInstance b, final double limit) {
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
        double[] row = getRow(0);
        double[] prevRow;
        
        // generate weights
        generateWeights(Math.max(aLength, bLength));
        
        // process top left cell of mat
        double min = row[j++] = weights[j] * cost(a, i, b, j);
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
        ParamHandlerUtils.setParam(param, WDTW.G_FLAG, this::setG, Double::parseDouble);
    }
}
