package tsml.classifiers.distance_based.utils.stats.scoring.v2;

import org.junit.Assert;
import org.junit.Test;

import java.util.Arrays;

public class GiniEntropyTest {
    
    @Test
    public void testPure() {
        Assert.assertEquals(0.6666666666666667, new GiniEntropy().score(new Labels<>(Arrays.asList(0,0,0,0,1,1,1,1,2,2,2,2)), Arrays.asList(new Labels<>(Arrays.asList(0,0,0,0)), new Labels<>(Arrays.asList(1,1,1,1)), new Labels<>(Arrays.asList(2,2,2,2)))), 0d);
    }

    @Test
    public void testImpure() {
        Assert.assertEquals(0.020000000000000018, new GiniEntropy().score(new Labels<>(Arrays.asList(0,0,0,0,1,1,1,1,1,1)), Arrays.asList(new Labels<>(Arrays.asList(0,0,1,1,1)), new Labels<>(Arrays.asList(0,0,1,1,1)))), 0d);
    }
    
    @Test
    public void testImpure2() {
        Assert.assertEquals(0.125, new GiniEntropy().score(new Labels<>(Arrays.asList(0,0,0,0,1,1,1,1)), Arrays.asList(new Labels<>(Arrays.asList(0,0,1,0)), new Labels<>(Arrays.asList(0,1,1,1)))), 0d);
    }

    @Test
    public void testA() {
        Assert.assertEquals(0.06666666666666674, new GiniEntropy().score(new Labels<>(Arrays.asList(0,0,0,0,1,1,1,1,1,1,1,1)), Arrays.asList(new Labels<>(Arrays.asList(0,0,0,1,1,1,1,1,1,1)), new Labels<>(Arrays.asList(0,1)))), 0d);
    }

    @Test
    public void testB() {
        Assert.assertEquals(0.22916666666666666, new GiniEntropy().score(new Labels<>(Arrays.asList(0,0,0,0,1,1,1,1,1,1,1,1)), Arrays.asList(new Labels<>(Arrays.asList(0,1,1,1,1,1,1,1)), new Labels<>(Arrays.asList(0,0,0,1)))), 0d);
    }
}
