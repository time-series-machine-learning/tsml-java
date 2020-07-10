package tsml.classifiers.distance_based.distances.ed;

import static tsml.classifiers.distance_based.distances.dtw.DTWDistanceTest.buildInstances;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import tsml.classifiers.distance_based.distances.dtw.DTWDistance;
import weka.core.Instances;

public class EDistanceTest {
    @Test
    public void matchesDtwZeroWindow() {
        DTWDistance dtw = new DTWDistance();
        dtw.setWindowSize(0);
        final Instances instances = buildInstances();
        dtw.setInstances(instances);
        final double d1 = df.distance(instances.get(0), instances.get(1));
        final double d2 = dtw.distance(instances.get(0), instances.get(1));
        Assert.assertEquals(d1, d2, 0d);
    }


    private Instances instances;
    private EDistance df;

    @Before
    public void before() {
        instances = buildInstances();
        df = new EDistance();
        df.setInstances(instances);
    }
}
