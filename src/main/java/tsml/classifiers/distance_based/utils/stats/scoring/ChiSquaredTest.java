package tsml.classifiers.distance_based.utils.stats.scoring;

import org.junit.Assert;
import org.junit.Test;

import java.util.Arrays;

public class ChiSquaredTest {

    @Test
    public void testImpure() {
        Assert.assertEquals(0, new ChiSquared().score(new Labels<>(Arrays.asList(0,0,0,0,1,1,1,1,1,1,1,1)), Arrays.asList(new Labels<>(Arrays.asList(0,0,1,1,1,1)), new Labels<>(Arrays.asList(0,0,1,1,1,1)))), 0d);
    }

    @Test
    public void testPure() {
        Assert.assertEquals(12.000000000000002, new ChiSquared().score(new Labels<>(Arrays.asList(0,0,0,0,1,1,1,1,1,1,1,1)), Arrays.asList(new Labels<>(Arrays.asList(0,0,0,0)), new Labels<>(Arrays.asList(1,1,1,1,1,1,1,1)))), 0d);
    }

    @Test
    public void testA() {
        Assert.assertEquals(4.6875, new ChiSquared().score(new Labels<>(Arrays.asList(0,0,0,0,0,0,0,0,1,1,1,1)), Arrays.asList(new Labels<>(Arrays.asList(0,0,0,0,0,0,0,1)), new Labels<>(Arrays.asList(0,1,1,1)))), 0d);
    }

    @Test
    public void testB() {
        Assert.assertEquals(0.30000000000000004, new ChiSquared().score(new Labels<>(Arrays.asList(0,0,0,0,1,1,1,1,1,1,1,1)), Arrays.asList(new Labels<>(Arrays.asList(0,0,0,1,1,1,1,1,1,1)), new Labels<>(Arrays.asList(0,1)))), 0d);
    }
}
