package tsml.classifiers.distance_based.utils.memory;

import org.junit.Assert;
import org.junit.Test;

public enum MemoryUnit {
    BYTES,
    KIBIBYTES,
    MEBIBYTES,
    GIBIBYTES,
    ;

    public static long FACTOR = 1024;

    public long convert(long amount, MemoryUnit unit) {
        int factorDifference = ordinal() - unit.ordinal();
        if(factorDifference > 0) {
            return amount / (FACTOR * factorDifference);
        } else {
            return amount * (FACTOR * -factorDifference);
        }
    }

    public long toBytes(long amount) {
        return convert(amount, BYTES);
    }

    public long toKibibytes(long amount) {
        return convert(amount, KIBIBYTES);
    }

    public long toMebibytes(long amount) {
        return convert(amount, MEBIBYTES);
    }

    public long toGibibytes(long amount) {
        return convert(amount, GIBIBYTES);
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
