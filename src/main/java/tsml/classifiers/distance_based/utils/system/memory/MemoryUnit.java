package tsml.classifiers.distance_based.utils.system.memory;

import org.junit.Assert;
import org.junit.Test;

public enum MemoryUnit {
    BYTES(1),
    B(BYTES),
    KIBIBYTES(1024, BYTES),
    KB(KIBIBYTES),
    KILOBYTES(1000, BYTES),
    MEBIBYTES(1024, KIBIBYTES),
    MB(MEBIBYTES),
    MEGABYTES(1000, KILOBYTES),
    GIBIBYTES(1024, MEBIBYTES),
    GB(GIBIBYTES),
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

    public static class UnitTests {
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
}
