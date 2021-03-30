package tsml.classifiers.distance_based.utils.stats.scoring;

import org.junit.Assert;
import org.junit.Test;

import java.util.Arrays;

import static tsml.classifiers.distance_based.utils.stats.scoring.Labels.fromCounts;

public class GiniGainTest {
    
    @Test
    public void testImpure() {
        Assert.assertEquals(0d, new GiniGain().score(fromCounts(Arrays.asList(6d, 6d)), Arrays.asList(fromCounts(Arrays.asList(3d, 3d)), fromCounts(Arrays.asList(3d, 3d)))), 0d);
    }
    
    @Test
    public void testPure() {
        Assert.assertEquals(0.5, new GiniGain().score(fromCounts(Arrays.asList(6d, 6d)), Arrays.asList(fromCounts(Arrays.asList(6d, 0d)), fromCounts(Arrays.asList(0d, 6d)))), 0d);
    }
    
    @Test
    public void testA() {
        Assert.assertEquals(0d, new GiniGain().score(fromCounts(Arrays.asList(8d, 4d)), Arrays.asList(fromCounts(Arrays.asList(4d, 2d)), fromCounts(Arrays.asList(4d, 2d)))), 0d);
    }

    @Test
    public void testB() {
        Assert.assertEquals(0.011111111111111072, new GiniGain().score(fromCounts(Arrays.asList(8d, 4d)), Arrays.asList(fromCounts(Arrays.asList(7d, 3d)), fromCounts(Arrays.asList(1d, 1d)))), 0d);
    }

    @Test
    public void testC() {
        Assert.assertEquals(0.1736111111111111, new GiniGain().score(fromCounts(Arrays.asList(8d, 4d)), Arrays.asList(fromCounts(Arrays.asList(7d, 1d)), fromCounts(Arrays.asList(1d, 3d)))), 0d);
    }
}
