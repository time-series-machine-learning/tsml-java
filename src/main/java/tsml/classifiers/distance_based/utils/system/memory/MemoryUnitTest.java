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
 
package tsml.classifiers.distance_based.utils.system.memory;

import static tsml.classifiers.distance_based.utils.system.memory.MemoryUnit.GIBIBYTES;
import static tsml.classifiers.distance_based.utils.system.memory.MemoryUnit.MEBIBYTES;

import org.junit.Assert;
import org.junit.Test;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class MemoryUnitTest {

    @Test
    public void gibibyteToMebibyte() {
        long amount = MEBIBYTES.convert(8, GIBIBYTES);
        Assert.assertEquals(amount, 8192);
    }

    @Test
    public void mebibyteToGibibyte() {
        long amount = GIBIBYTES.convert(8192, MEBIBYTES);
        Assert.assertEquals(amount, 8);
    }
}
