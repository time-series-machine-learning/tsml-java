package tsml.classifiers.distance_based.utils.stats.scoring;

import org.junit.Assert;
import org.junit.Test;

import java.util.Arrays;

import static tsml.classifiers.distance_based.utils.stats.scoring.Labels.fromCounts;

public class InfoEntropyTest {
    @Test
    public void testImpure() {
        Assert.assertEquals(1d, new InfoEntropy().entropy(new Labels<>().setDistribution(Arrays.asList(0.5, 0.5))), 0d);
    }

    @Test
    public void testPure() {
        Assert.assertEquals(0.0, new InfoEntropy().entropy(new Labels<>().setDistribution(Arrays.asList(1d, 0d))), 0d);
    }

    @Test
    public void testA() {
        Assert.assertEquals(0.8904916402194913, new InfoEntropy().entropy(new Labels<>().setDistribution(Arrays.asList(9d/13, 4d/13))), 0d);
    }

    @Test
    public void testB() {
        Assert.assertEquals(0.8812908992306928, new InfoEntropy().entropy(new Labels<>().setDistribution(Arrays.asList(7d/10, 3d/10))), 0d);
    }


    @Test
    public void testImpureScore() {
        Assert.assertEquals(0d, new InfoEntropy().score(fromCounts(Arrays.asList(6d, 6d)), Arrays.asList(fromCounts(Arrays.asList(3d, 3d)), fromCounts(Arrays.asList(3d, 3d)))), 0d);
    }

    @Test
    public void testPureScore() {
        Assert.assertEquals(1d, new InfoEntropy().score(fromCounts(Arrays.asList(6d, 6d)), Arrays.asList(fromCounts(Arrays.asList(6d, 0d)), fromCounts(Arrays.asList(0d, 6d)))), 0d);
    }

    @Test
    public void testAScore() {
        Assert.assertEquals(0.08170416594551044, new InfoEntropy().score(fromCounts(Arrays.asList(8d, 4d)), Arrays.asList(fromCounts(Arrays.asList(4d, 2d)), fromCounts(Arrays.asList(4d, 2d)))), 0d);
    }

    @Test
    public void testBScore() {
        Assert.assertEquals(0.09892425064108933, new InfoEntropy().score(fromCounts(Arrays.asList(8d, 4d)), Arrays.asList(fromCounts(Arrays.asList(7d, 3d)), fromCounts(Arrays.asList(1d, 1d)))), 0d);
    }

    @Test
    public void testCScore() {
        Assert.assertEquals(0.36719766304722473, new InfoEntropy().score(fromCounts(Arrays.asList(8d, 4d)), Arrays.asList(fromCounts(Arrays.asList(7d, 1d)), fromCounts(Arrays.asList(1d, 3d)))), 0d);
    }
}
