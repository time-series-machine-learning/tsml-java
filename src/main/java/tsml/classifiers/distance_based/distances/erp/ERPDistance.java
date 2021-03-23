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
 
package tsml.classifiers.distance_based.distances.erp;

import tsml.classifiers.distance_based.distances.MatrixBasedDistanceMeasure;
import tsml.classifiers.distance_based.distances.dtw.DTW;
import tsml.classifiers.distance_based.utils.collections.checks.Checks;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import tsml.data_containers.TimeSeries;
import tsml.data_containers.TimeSeriesInstance;
import utilities.Utilities;

import java.util.Arrays;

/**
 * ERP distance measure.
 * <p>
 * Contributors: goastler
 */
public class ERPDistance extends MatrixBasedDistanceMeasure {

    public static final String WINDOW_FLAG = DTW.WINDOW_FLAG;
    public static final String G_FLAG = "g";
    private double g = 0.01;
    private double window = 1;
    
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
            final double aValue = aDim.get(aIndex);
            final double sqDiff = StrictMath.pow(aValue - g, 2);
            sum += sqDiff;
        }
        return sum;
    }
    
    public double cost(TimeSeriesInstance a, int aIndex, TimeSeriesInstance b, int bIndex) {
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
        final double windowSize = this.window * bLength;

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
        return super.getParams().add(DTW.WINDOW_FLAG, window).add(G_FLAG, g);
    }

    @Override
    public void setParams(final ParamSet param) throws Exception {
        super.setParams(param);
        setG(param.get(G_FLAG, getG()));
        setWindow(param.get(WINDOW_FLAG, getWindow()));
    }

    public double getWindow() {
        return window;
    }

    public void setWindow(final double window) {
        this.window = Checks.requireUnitInterval(window);
    }
}
