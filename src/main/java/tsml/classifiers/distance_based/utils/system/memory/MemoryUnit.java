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
