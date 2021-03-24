/*
 * This file is part of the UEA Time Series Machine Learning (TSML) toolbox.
 *
 * The UEA TSML toolbox is free software: you can redistribute it and/or 
 * modify it under the terms of the GNU General Public License as published 
 * by the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version.
 *
 * The UEA TSML toolbox is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with the UEA TSML toolbox. If not, see <https://www.gnu.org/licenses/>.
 */
 
package tsml.classifiers.distance_based.distances.twed;

import tsml.classifiers.distance_based.distances.MatrixBasedDistanceMeasure;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import tsml.data_containers.TimeSeries;
import tsml.data_containers.TimeSeriesInstance;
import utilities.ArrayUtilities;

import java.util.Arrays;

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
            final double aValue = aDim.get(aIndex);
            final double bValue = bDim.get(bIndex);
            final double sqDiff = StrictMath.pow(aValue - bValue, 2);
            sum += sqDiff;
        }
        return sum;
    }
    
    private double cellCost(final TimeSeriesInstance a, final int aIndex) {
        double sum = 0;
        for(int i = 0; i < a.getNumDimensions(); i++) {
            final TimeSeries aDim = a.get(i);
            final double aValue = aDim.get(aIndex);
            final double sq = StrictMath.pow(aValue, 2);
            sum += sq;
        }
        return sum;
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
        setup(aLength + 1, bLength + 1, true);

        // step is the increment of the mid point for each row
        final double step = (double) (bLength) / (aLength);
        final double windowSize = 1d * bLength + 1; // +1 because of padding col

        // row index
        int i = 0;

        // start and end of window
        int start = 0;
        double mid = 0;
        int end = Math.min(bLength, (int) Math.floor(windowSize)); // +1 as matrix padded by 1 row and 1 col
        int prevEnd; // store end of window from previous row to fill in shifted space with inf
        double[] row = getRow(i);
        double[] prevRow;
        double[] jCosts = new double[bLength + 1];
        double min, iCost;

        // col index
        int j = start;

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
        i++;

        // process remaining rows
        for(; i < aLength + 1; i++) {

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
            end = Math.min(bLength, (int) Math.floor(mid + windowSize));
            j = start;

            // set the values above the current row and outside of previous window to inf
            Arrays.fill(prevRow, prevEnd + 1, end + 1, Double.POSITIVE_INFINITY);
            // set the value left of the window to inf
            if(j > 0) row[j - 1] = Double.POSITIVE_INFINITY; // >1 as matrix padded with 1 row and 1 col

            // fill any jCosts which have not yet been visited
            for(int x = prevEnd + 1; x <= end; x++) {
                jCosts[x] = cost(b, x - 2, b, x - 1);
            }

            // the ith cost for this row
            if(i > 1) {
                iCost = cost(a, i - 2, a, i - 1);
            } else {
                iCost = cellCost(a, i - 1);
            }

            // if assessing the left most column then only mapping option is top - not left or topleft
            if(j == 0) {
                row[j] = prevRow[j] + iCost;
                min = Math.min(min, row[j++]);
            }

            // compute the distance for each cell in the row
            for(; j <= end; j++) {
                double dist = cost(a, i - 1, b, j - 1);
                double htrans = Math.abs(i - j);
                if(i > 1 && j > 1) {
                    dist += cost(a, i - 2, b, j - 2);
                    htrans *= 2;
                }
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
        setLambda(param.get(LAMBDA_FLAG, getLambda()));
        setNu(param.get(NU_FLAG, getNu()));
    }

    public static void main(String[] args) {
        final TWEDistance dm = new TWEDistance();
        dm.setRecordCostMatrix(true);
        System.out.println(dm.distance(new TimeSeriesInstance(new double[][]{{1,2,3,3,2,4,5,2}}), new TimeSeriesInstance(new double[][] {{2,3,4,5,6,2,2,2}})));
        System.out.println(ArrayUtilities.toString(dm.costMatrix()));
    }

}
