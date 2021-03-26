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
import org.junit.Before;
import org.junit.Test;

public class IntIntervalTest {

    private IntInterval interval;
    private int start;
    private int length;

    @Before
    public void before() {
        interval = new IntInterval();
        this.start = 50;
        this.length = 11;
        interval.setStart(start);
        interval.setEnd(length + start - 1);
    }

    @Test
    public void testIntervalSize() {
        Assert.assertEquals(length, (long) interval.size());
    }

    @Test
    public void testTranslate() {
        for(int i = 0; i < length; i++) {
            final int index = interval.translate(i);
            Assert.assertEquals(i + start, index);
        }
    }

    @Test
    public void testInverseTranslate() {
        for(int i = 0; i < length; i++) {
            final int index = interval.inverseTranslate(i + start);
            Assert.assertEquals(i, index);
        }
    }

    @Test(expected = ArrayIndexOutOfBoundsException.class)
    public void testInverseTranslateOutOfBoundsAbove() {
        interval.inverseTranslate(start + length);
    }

    @Test(expected = ArrayIndexOutOfBoundsException.class)
    public void testInverseTranslateOutOfBoundsBelow() {
        interval.inverseTranslate(start - 1);
    }


    @Test(expected = ArrayIndexOutOfBoundsException.class)
    public void testTranslateOutOfBoundsAbove() {
        interval.translate(length);
    }

    @Test(expected = ArrayIndexOutOfBoundsException.class)
    public void testTranslateOutOfBoundsBelow() {
        interval.translate(-1);
    }
}
