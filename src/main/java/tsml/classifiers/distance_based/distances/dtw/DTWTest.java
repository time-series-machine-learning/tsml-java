package tsml.classifiers.distance_based.distances.dtw;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import utilities.InstanceTools;
import weka.core.Instances;

/**
 * Purpose: test dtw
 * <p>
 * Contributors: goastler
 */
public class DTWTest {

    public static Instances buildInstances() {
        return InstanceTools.toWekaInstancesWithClass(new double[][] {
            {1,2,3,4,5,0},
            {6,11,15,2,7,1}
        });
    }

    private Instances instances;
    private DTWDistance df;

    @Before
    public void before() {
        instances = buildInstances();
        df = new DTWDistance();
        df.setInstances(instances);
    }

    @Test
    public void testDtwFullWarp() {
        df.setWarpingWindow(-1);
        Assert.assertFalse(df.isWarpingWindowInPercentage());
        double distance = df.distance(instances.get(0), instances.get(1));
        Assert.assertEquals(distance, 203, 0);
    }

    @Test
    public void testDtwFullWarpPercentage() {
        df.setWarpingWindowPercentage(-1);
        Assert.assertTrue(df.isWarpingWindowInPercentage());
        double distance = df.distance(instances.get(0), instances.get(1));
        Assert.assertEquals(distance, 203, 0);
    }

    @Test
    public void testDtwConstrainedWarp() {
        df.setWarpingWindow(2);
        Assert.assertFalse(df.isWarpingWindowInPercentage());
        double distance = df.distance(instances.get(0), instances.get(1));
        Assert.assertEquals(distance, 212, 0);
    }

    @Test
    public void testDtwConstrainedWarpPercentage() {
        df.setWarpingWindowPercentage(0.5);
        Assert.assertTrue(df.isWarpingWindowInPercentage());
        double distance = df.distance(instances.get(0), instances.get(1));
        Assert.assertEquals(distance, 212, 0);
    }
}
