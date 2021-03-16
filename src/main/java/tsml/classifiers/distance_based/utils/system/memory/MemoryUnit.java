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

import org.junit.Assert;
import org.junit.Test;

public enum MemoryUnit {
    BYTES(1),
    KIBIBYTES(1024, BYTES),
    KILOBYTES(1000, BYTES),
    MEBIBYTES(1024, KIBIBYTES),
    MEGABYTES(1000, KILOBYTES),
    GIBIBYTES(1024, MEBIBYTES),
    GIGABYTES(1000, MEGABYTES),
    ;

    private final long oneUnitInBytes;

    MemoryUnit(final long oneUnitInBytes) {
        Assert.assertTrue(oneUnitInBytes > 0);
        this.oneUnitInBytes = oneUnitInBytes;
    }

    MemoryUnit(MemoryUnit alias) {
        this(1, alias);
    }

    MemoryUnit(long amount, MemoryUnit unit) {
        this(amount * unit.oneUnitInBytes);
    }

    public long convert(long amount, MemoryUnit unit) {
        if(oneUnitInBytes > unit.oneUnitInBytes) {
            long ratio = oneUnitInBytes / unit.oneUnitInBytes;
            return amount / ratio;
        } else {
            long ratio = unit.oneUnitInBytes / oneUnitInBytes;
            return amount * ratio;
        }
    }
}
