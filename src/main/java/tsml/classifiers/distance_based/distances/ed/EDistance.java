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
import weka.core.Instance;

public class EDistance extends BaseDistanceMeasure {

    public double findDistance(final Instance a, Instance b, final double limit) {
        double sum = 0;

        int aLength = a.numAttributes() - 1;
        int bLength = b.numAttributes() - 1;

        for(int i = 0; i < aLength; i++) {
            sum += Math.pow(a.value(i) - b.value(i), 2);
            if(sum > limit) {
                return Double.POSITIVE_INFINITY;
            }
        }

        return sum;
    }
}
