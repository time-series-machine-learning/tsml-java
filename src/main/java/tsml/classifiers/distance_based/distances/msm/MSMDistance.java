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
 
package tsml.classifiers.distance_based.distances.msm;

import tsml.classifiers.distance_based.distances.MatrixBasedDistanceMeasure;
import tsml.classifiers.distance_based.utils.collections.params.ParamHandlerUtils;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import tsml.data_containers.TimeSeries;
import tsml.data_containers.TimeSeriesInstance;

import java.util.Arrays;

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
        final double windowSize = 1d * bLength;

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
        setC(param.get(C_FLAG, getC()));
    }
}
