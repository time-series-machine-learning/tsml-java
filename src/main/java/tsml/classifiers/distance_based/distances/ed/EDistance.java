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
 
package tsml.classifiers.distance_based.distances.ed;

import tsml.classifiers.distance_based.distances.BaseDistanceMeasure;
import tsml.classifiers.distance_based.distances.MatrixBasedDistanceMeasure;
import tsml.classifiers.distance_based.distances.dtw.DTWDistance;
import tsml.data_containers.TimeSeriesInstance;

public class EDistance extends BaseDistanceMeasure {
    
    public double distance(final TimeSeriesInstance a, TimeSeriesInstance b, final double limit) {
        double sum = 0;

        final int aLength = a.getMaxLength();
        final int bLength = b.getMaxLength();

        for(int i = 0; i < aLength; i++) {
            sum += DTWDistance.cost(a, i, b, i);
            if(sum > limit) {
                return Double.POSITIVE_INFINITY;
            }
        }

        return sum;
    }
}
