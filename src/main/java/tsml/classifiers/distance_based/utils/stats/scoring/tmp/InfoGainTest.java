package tsml.classifiers.distance_based.utils.stats.scoring.tmp;

import org.junit.Assert;
import org.junit.Test;

import java.util.Arrays;

public class InfoGainTest {

    @Test
    public void testPure() {
        Assert.assertEquals(1.584962500721156, new InfoGain().score(new Labels<>(Arrays.asList(0,0,0,0,1,1,1,1,2,2,2,2)), Arrays.asList(new Labels<>(Arrays.asList(0,0,0,0)), new Labels<>(Arrays.asList(1,1,1,1)), new Labels<>(Arrays.asList(2,2,2,2)))), 0d);
    }

    @Test
    public void testImpure() {
        Assert.assertEquals(0, new InfoGain().score(new Labels<>(Arrays.asList(0,0,0,0,1,1,1,1,1,1,1,1)), Arrays.asList(new Labels<>(Arrays.asList(0,0,1,1,1,1)), new Labels<>(Arrays.asList(0,0,1,1,1,1)))), 0d);
    }

    @Test
    public void testImpure2() {
        Assert.assertEquals(0.18872187554086717, new InfoGain().score(new Labels<>(Arrays.asList(0,0,0,0,1,1,1,1)), Arrays.asList(new Labels<>(Arrays.asList(0,0,1,0)), new Labels<>(Arrays.asList(0,1,1,1)))), 0d);
    }

    @Test
    public void testA() {
        Assert.assertEquals(0.017220084695579008, new InfoGain().score(new Labels<>(Arrays.asList(0,0,0,0,1,1,1,1,1,1,1,1)), Arrays.asList(new Labels<>(Arrays.asList(0,0,0,1,1,1,1,1,1,1)), new Labels<>(Arrays.asList(0,1)))), 0d);
    }

    @Test
    public void testB() {
        Assert.assertEquals(0.28549349710171434, new InfoGain().score(new Labels<>(Arrays.asList(0,0,0,0,1,1,1,1,1,1,1,1)), Arrays.asList(new Labels<>(Arrays.asList(0,1,1,1,1,1,1,1)), new Labels<>(Arrays.asList(0,0,0,1)))), 0d);
    }
}
