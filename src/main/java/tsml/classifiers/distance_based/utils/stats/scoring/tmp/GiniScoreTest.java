package tsml.classifiers.distance_based.utils.stats.scoring.tmp;

import org.junit.Assert;
import org.junit.Test;

import java.util.Arrays;

public class GiniScoreTest {

    @Test
    public void testPure() {
        Assert.assertEquals(0.6666666666666667, new GiniScore().score(new Labels<>(Arrays.asList(0,0,0,0,1,1,1,1,2,2,2,2)), Arrays.asList(new Labels<>(Arrays.asList(0,0,0,0)), new Labels<>(Arrays.asList(1,1,1,1)), new Labels<>(Arrays.asList(2,2,2,2)))), 0d);
    }

    @Test
    public void testImpure() {
        Assert.assertEquals(0, new GiniScore().score(new Labels<>(Arrays.asList(0,0,0,0,1,1,1,1,1,1,1,1)), Arrays.asList(new Labels<>(Arrays.asList(0,0,1,1,1,1)), new Labels<>(Arrays.asList(0,0,1,1,1,1)))), 0d);
    }
    
    @Test
    public void testA() {
        Assert.assertEquals(0.011111111111111127, new GiniScore().score(new Labels<>(Arrays.asList(0,0,0,0,1,1,1,1,1,1,1,1)), Arrays.asList(new Labels<>(Arrays.asList(0,0,0,1,1,1,1,1,1,1)), new Labels<>(Arrays.asList(0,1)))), 0d);
    }
    
    @Test
    public void testB() {
        Assert.assertEquals(0.1736111111111111, new GiniScore().score(new Labels<>(Arrays.asList(0,0,0,0,1,1,1,1,1,1,1,1)), Arrays.asList(new Labels<>(Arrays.asList(0,1,1,1,1,1,1,1)), new Labels<>(Arrays.asList(0,0,0,1)))), 0d);
    }


    @Test
    public void testEntropyPure() {
        final Labels<Integer> labels = new Labels<>(Arrays.asList(5, 5, 5, 5, 5));
        labels.setLabelSet(Arrays.asList(5,1));
        Assert.assertEquals(0, new GiniScore().entropy(labels), 0d);
    }

    @Test
    public void testEntropyImpure() {
        final Labels<Integer> labels = new Labels<>(Arrays.asList(1,2,1,2,1,2));
        labels.setLabelSet(Arrays.asList(1,2));
        Assert.assertEquals(0.5, new GiniScore().entropy(labels), 0d);
    }

    @Test
    public void testEntropyA() {
        final Labels<Integer> labels = new Labels<>(Arrays.asList(0,0,0,0,1,1,1,1,1,1));
        labels.setLabelSet(Arrays.asList(1,0));
        Assert.assertEquals(0.48, new GiniScore().entropy(labels), 0d);
    }

    @Test
    public void testEntropyB() {
        final Labels<Integer> labels = new Labels<>(Arrays.asList(0,0,0,0,0,0,0,0,0,1));
        labels.setLabelSet(Arrays.asList(1,0));
        Assert.assertEquals(0.17999999999999994, new GiniScore().entropy(labels), 0d);
    }

    @Test
    public void testEntropyC() {
        final Labels<Integer> labels = new Labels<>(Arrays.asList(0,0,0,0,1,1,1,1,1,1,1,1));
        labels.setLabelSet(Arrays.asList(1,0));
        Assert.assertEquals(0.4444444444444444, new GiniScore().entropy(labels), 0d);
    }

}
