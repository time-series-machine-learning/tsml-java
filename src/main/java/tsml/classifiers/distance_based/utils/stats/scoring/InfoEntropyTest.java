package tsml.classifiers.distance_based.utils.stats.scoring;

import org.junit.Assert;
import org.junit.Test;

import java.util.Arrays;

public class InfoEntropyTest {

    @Test
    public void testPure() {
        Assert.assertEquals(1.5849625007211559, new InfoEntropy().score(new Labels<>(
                Arrays.asList(0,0,0,0,1,1,1,1,2,2,2,2)), Arrays.asList(new Labels<>(Arrays.asList(0,0,0,0)), new Labels<>(Arrays.asList(1,1,1,1)), new Labels<>(Arrays.asList(2,2,2,2)))), 0d);
    }

    @Test
    public void testImpure() {
        Assert.assertEquals(0, new InfoEntropy().score(new Labels<>(Arrays.asList(0,0,0,0,0,0,1,1,1,1,1,1)), Arrays.asList(new Labels<>(Arrays.asList(0,0,0,1,1,1)), new Labels<>(Arrays.asList(0,0,0,1,1,1)))), 0d);
    }

    @Test
    public void testImpure2() {
        Assert.assertEquals(0.18872187554086717, new InfoEntropy().score(new Labels<>(Arrays.asList(0,0,0,0,1,1,1,1)), Arrays.asList(new Labels<>(Arrays.asList(0,0,1,0)), new Labels<>(Arrays.asList(0,1,1,1)))), 0d);
    }

    @Test
    public void testA() {
        Assert.assertEquals(0.09892425064108942, new InfoEntropy().score(new Labels<>(Arrays.asList(0,0,0,0,1,1,1,1,1,1,1,1)), Arrays.asList(new Labels<>(Arrays.asList(0,0,0,1,1,1,1,1,1,1)), new Labels<>(Arrays.asList(0,1)))), 0d);
    }

    @Test
    public void testB() {
        Assert.assertEquals(0.36719766304722473, new InfoEntropy().score(new Labels<>(Arrays.asList(0,0,0,0,1,1,1,1,1,1,1,1)), Arrays.asList(new Labels<>(Arrays.asList(0,1,1,1,1,1,1,1)), new Labels<>(Arrays.asList(0,0,0,1)))), 0d);
    }
}
