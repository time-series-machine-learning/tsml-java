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
