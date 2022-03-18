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
 
package tsml.classifiers.distance_based.utils.collections.intervals;

import org.junit.Assert;
import org.junit.Test;
import weka.core.DenseInstance;

public class IntervalInstanceTest {

    @Test
    public void testIndices() {
        final double[] attributes = new double[10];
        for(int i = 0; i < attributes.length; i++) {
            attributes[i] = i;
        }
        final DenseInstance instance = new DenseInstance(1, attributes);
        int start = 5;
        int length = 4;
        final IntervalInstance intervalInstance = new IntervalInstance(new IntInterval(start, length), instance);
        for(int i = 0; i < intervalInstance.numAttributes() - 1; i++) {
            final double intervalValue = intervalInstance.value(i);
            final double instanceValue = instance.value(i + start);
            Assert.assertEquals(intervalValue, instanceValue, 0);
        }
    }
}
