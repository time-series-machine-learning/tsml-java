package tsml.classifiers.distance_based.utils.stats.scoring.a;

import org.junit.Assert;
import org.junit.Test;
import tsml.classifiers.distance_based.utils.stats.scoring.v2.Labels;

import java.util.Arrays;

import static tsml.classifiers.distance_based.utils.stats.scoring.v2.Labels.fromCounts;

public class GiniEntropyTest {
    @Test
    public void testImpure() {
        Assert.assertEquals(0.5, new GiniEntropy().entropy(new Labels<>().setDistribution(Arrays.asList(0.5, 0.5))), 0d);
    }

    @Test
    public void testPure() {
        Assert.assertEquals(0.0, new GiniEntropy().entropy(new Labels<>().setDistribution(Arrays.asList(1d, 0d))), 0d);
    }

    @Test
    public void testA() {
        Assert.assertEquals(0.48, new GiniEntropy().entropy(new Labels<>().setDistribution(Arrays.asList(0.4, 0.6))), 0d);
    }

    @Test
    public void testB() {
        Assert.assertEquals(0.17999999999999994, new GiniEntropy().entropy(new Labels<>().setDistribution(Arrays.asList(0.1, 0.9))), 0d);
    }


    @Test
    public void testImpureScore() {
        Assert.assertEquals(0d, new GiniEntropy().score(fromCounts(Arrays.asList(6d, 6d)), Arrays.asList(fromCounts(Arrays.asList(3d, 3d)), fromCounts(Arrays.asList(3d, 3d)))), 0d);
    }

    @Test
    public void testPureScore() {
        Assert.assertEquals(0.5, new GiniEntropy().score(fromCounts(Arrays.asList(6d, 6d)), Arrays.asList(fromCounts(Arrays.asList(6d, 0d)), fromCounts(Arrays.asList(0d, 6d)))), 0d);
    }

    @Test
    public void testAScore() {
        Assert.assertEquals(0.05555555555555558, new GiniEntropy().score(fromCounts(Arrays.asList(8d, 4d)), Arrays.asList(fromCounts(Arrays.asList(4d, 2d)), fromCounts(Arrays.asList(4d, 2d)))), 0d);
    }

    @Test
    public void testBScore() {
        Assert.assertEquals(0.06666666666666664, new GiniEntropy().score(fromCounts(Arrays.asList(8d, 4d)), Arrays.asList(fromCounts(Arrays.asList(7d, 3d)), fromCounts(Arrays.asList(1d, 1d)))), 0d);
    }

    @Test
    public void testCScore() {
        Assert.assertEquals(0.22916666666666666, new GiniEntropy().score(fromCounts(Arrays.asList(8d, 4d)), Arrays.asList(fromCounts(Arrays.asList(7d, 1d)), fromCounts(Arrays.asList(1d, 3d)))), 0d);
    }
}
