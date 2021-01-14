package tsml.classifiers.distance_based.distances.twed;

import tsml.classifiers.distance_based.distances.MatrixBasedDistanceMeasure;
import tsml.classifiers.distance_based.utils.collections.params.ParamHandlerUtils;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import tsml.data_containers.TimeSeries;
import tsml.data_containers.TimeSeriesInstance;

import java.util.Arrays;

import static utilities.ArrayUtilities.*;

/**
 * TWED distance measure.
 * <p>
 * Contributors: goastler
 */
public class TWEDistance
    extends MatrixBasedDistanceMeasure {

    private double lambda = 1;
    private double nu = 1;

    public static final String NU_FLAG = "n";
    public static final String LAMBDA_FLAG = "l";

    private double cost(final TimeSeriesInstance a, final int aIndex, final TimeSeriesInstance b, final int bIndex) {
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
    
    private double cellCost(final TimeSeriesInstance a, final int aIndex) {
        double sum = 0;
        for(int i = 0; i < a.getNumDimensions(); i++) {
            final TimeSeries aDim = a.get(i);
            final Double aValue = aDim.get(aIndex);
            final double sq = Math.pow(aValue, 2);
            sum += sq;
        }
        return sum;
    }
    
    @Override
    public double distance(final TimeSeriesInstance a, final TimeSeriesInstance b, final double limit) {
        
        final int aLength = a.getMaxLength();
        final int bLength = b.getMaxLength();
        setup(aLength + 1, bLength + 1, true);
        final double lengthRatio = (double) bLength / aLength;
        final double windowSize = 1d * bLength + 1; // +1 as the matrix is padded with 1 row and 1 col
        
        // start and end of window
        int i = 0; // start at first row
        double mid = i * lengthRatio;
        int start = 0;
        int end = (int) Math.min(bLength, Math.ceil(mid + windowSize)); // +1 as matrix padded by 1 row and 1 col
        int prevEnd;
        int j = start;
        double[] row = getRow(0);
        double[] prevRow;
        double[] jCosts = new double[bLength + 1];
        
        // border of the cost matrix initialization
        row[j++] = 0;
        row[j] = jCosts[j] = cellCost(b, i);
        j++;
        // compute the first padded row
        for(; j <= end; j++) {
            //CHANGE AJB 8/1/16: Only use power of 2 for speed up
            jCosts[j] = cost(b, j - 2, b, j - 1);
            row[j] = row[j - 1] + jCosts[j];
        }
        
        // compute first row
        i++; // make i==1, i.e. point to the first row. The row before is the padding row
        prevEnd = end;
        mid = i * lengthRatio;
        j = start = (int) Math.max(0, Math.floor(mid - windowSize)); // +1 as start from the second cell - first cell filled manually below
        end = (int) Math.min(bLength, Math.ceil(mid + windowSize)); // +1 as matrix padded with 1 row and 1 col
        double iCost = cellCost(a, 0);
        
        // change rows
        prevRow = row;
        row = getRow(i);
        
        // set the top values outside of window to inf
        Arrays.fill(prevRow, prevEnd + 1, end + 1, Double.POSITIVE_INFINITY);
        // set the value left of the window to inf
        if(j > 0) row[j - 1] = Double.POSITIVE_INFINITY;
        
        // compute first cell in row
        double min = row[j++] = iCost;
        // compute remaining cells in the first row
        for(; j <= end; j++) {
            final double dist = cost(a, i - 1, b, j - 1);
            final double htrans = Math.abs((i - j));
            final double topLeft = prevRow[j - 1] + nu * htrans + dist;
            final double top = iCost + prevRow[j] + lambda + nu * i;
            final double left = jCosts[j] + row[j - 1] + lambda + nu;
            final double cost = Math.min(topLeft, Math.min(top, left));
            row[j] = cost;
            min = Math.min(min, cost);
        }
        if(min > limit) return Double.POSITIVE_INFINITY; // quit if beyond limit
        i++;
        
        // process remaining rows
        for(; i < aLength + 1; i++) {
            // reset min for the row
            min = Double.POSITIVE_INFINITY;
            
            // start, end and mid of window
            prevEnd = end;
            mid = i * lengthRatio;
            start = (int) Math.max(0, Math.floor(mid - windowSize));
            end = (int) Math.min(bLength, Math.ceil(mid + windowSize)); // +1 as matrix padded with 1 row and 1 col
            j = start;
            
            // change rows
            prevRow = row;
            row = getRow(i);
            
            // set the top values outside of window to inf
            Arrays.fill(prevRow, prevEnd + 1, end + 1, Double.POSITIVE_INFINITY);
            // set the value left of the window to inf
            if(j > 0) row[j - 1] = Double.POSITIVE_INFINITY;
            
            // fill any jCosts which have not yet been visited
            for(int x = prevEnd + 1; x <= end; x++) {
                jCosts[x] = cost(b, x - 2, b, x - 1);
            }
            
            // the ith cost for this row
            iCost = cost(a, i - 2, a, i - 1);
            
            // if assessing the left most column then only mapping option is top - not left or topleft
            if(j == 0) {
                row[j] = prevRow[j] + iCost;
                min = Math.min(min, row[j++]);
            }
            
            if(j == 1) {
                final double dist = cost(a, i - 1, b, 0);
                final double htrans = Math.abs(i - j);
                final double topLeft = prevRow[0] + nu * htrans + dist;
                final double top = iCost + prevRow[1] + lambda + nu;
                final double left = jCosts[1] + row[0] + lambda + nu;
                row[j] = Math.min(topLeft, Math.min(top, left));
                min = Math.min(min, row[j++]);
            }
            
            // compute the distance for each cell in the row
            for(; j <= end; j++) {
                final double dist = cost(a, i - 1, b, j - 1) + cost(a, i - 2, b, j - 2);
                final double htrans = Math.abs(i - j) * 2;
                final double topLeft = prevRow[j - 1] + nu * htrans + dist;
                final double top = iCost + prevRow[j] + lambda + nu;
                final double left = jCosts[j] + row[j - 1] + lambda + nu;
                row[j] = Math.min(topLeft, Math.min(top, left));
                min = Math.min(min, row[j]);
            }
            if(min > limit) return Double.POSITIVE_INFINITY; // quit if beyond limit
        }
        // last value in the current row is the distance
        final double distance = row[row.length - 1];
        teardown();
        return distance;
    }

    public double getLambda() {
        return lambda;
    }

    public void setLambda(double lambda) {
        this.lambda = lambda;
    }

    public double getNu() {
        return nu;
    }

    public void setNu(double nu) {
        this.nu = nu;
    }

    @Override
    public ParamSet getParams() {
        return super.getParams().add(NU_FLAG, nu).add(LAMBDA_FLAG, lambda);
    }

    @Override
    public void setParams(final ParamSet param) throws Exception {
        super.setParams(param);
        ParamHandlerUtils.setParam(param, NU_FLAG, this::setNu, Double::parseDouble);
        ParamHandlerUtils.setParam(param, LAMBDA_FLAG, this::setLambda, Double::parseDouble);
    }

    public static void main(String[] args) {
        final TWEDistance dm = new TWEDistance();
        dm.setGenerateDistanceMatrix(true);
        System.out.println(dm.distance(new TimeSeriesInstance(new double[][]{{1,2,3,3,2,4,5,2}}), new TimeSeriesInstance(new double[][] {{2,3,4,5,6,2,2,2}})));
    }

}
