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

import java.util.Locale;
import java.util.Objects;

public class MemoryUnit implements Comparable<MemoryUnit> {
    
    public static final MemoryUnit BYTES = new MemoryUnit(1);
    public static final MemoryUnit KIBIBYTES = new MemoryUnit(1024);
    public static final MemoryUnit KILOBYTES = new MemoryUnit(1000);
    public static final MemoryUnit MEBIBYTES = new MemoryUnit(1024, KIBIBYTES);
    public static final MemoryUnit MEGABYTES = new MemoryUnit(1000, KILOBYTES);
    public static final MemoryUnit GIGABYTES = new MemoryUnit(1000, MEGABYTES);
    public static final MemoryUnit GIBIBYTES = new MemoryUnit(1024, MEBIBYTES);

    private final long oneUnitInBytes;

    private MemoryUnit(final long oneUnitInBytes) {
        Assert.assertTrue(oneUnitInBytes > 0);
        this.oneUnitInBytes = oneUnitInBytes;
    }

    private MemoryUnit(MemoryUnit alias) {
        this(1, alias);
    }

    private MemoryUnit(long amount, MemoryUnit unit) {
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
    
    public static MemoryUnit valueOf(String str) {
        str = str.toLowerCase();
        switch(str) {
            case "": // default to MBs
            case "mb":
            case "mebibyte":
            case "mebibytes": return MEBIBYTES;
            case "b":
            case "bytes":
            case "byte": return BYTES;
            case "kb":
            case "kibibyte":
            case "kibibytes": return KIBIBYTES;
            case "gb":
            case "gibibyte":
            case "gibibytes": return GIBIBYTES;
            case "gigabyte":
            case "gigabytes": return GIGABYTES;
            case "kilobyte":
            case "kilobytes": return KILOBYTES;
            case "megabyte":
            case "megabytes": return MEGABYTES;
            default: throw new IllegalArgumentException("unknown memory unit type: " + str);
        }
    }

    @Override public int compareTo(final MemoryUnit other) {
        return Long.compare(oneUnitInBytes, other.oneUnitInBytes);
    }

    @Override public boolean equals(final Object o) {
        if(this == o) {
            return true;
        }
        if(!(o instanceof MemoryUnit)) {
            return false;
        }
        final MemoryUnit that = (MemoryUnit) o;
        return oneUnitInBytes == that.oneUnitInBytes;
    }

    @Override public int hashCode() {
        return Objects.hash(oneUnitInBytes);
    }
}
