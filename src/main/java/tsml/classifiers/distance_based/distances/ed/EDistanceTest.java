package tsml.classifiers.distance_based.distances.ed;

import static tsml.classifiers.distance_based.distances.dtw.DTWDistanceTest.*;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.junit.runners.Parameterized;
import tsml.classifiers.distance_based.distances.DistanceMeasureSpaceBuilder;
import tsml.classifiers.distance_based.distances.dtw.DTWDistance;
import tsml.classifiers.distance_based.distances.dtw.DTWDistanceTest;
import weka.core.Instances;

import java.util.Collection;

public class EDistanceTest {
    @Test
    public void matchesDtwZeroWindow() {
        DTWDistance dtw = new DTWDistance();
        dtw.setWindowSize(0);
        final Instances instances = buildInstances();
        dtw.buildDistanceMeasure(instances);
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
        df.buildDistanceMeasure(instances);
    }


    public static class TestOnDatasets extends DTWDistanceTest.TestOnDatasets {

        @Override public DistanceMeasureSpaceBuilder getBuilder() {
            return DistanceMeasureSpaceBuilder.ED;
        }

        @Parameterized.Parameters(name = "{0}")
        public static Collection<Object[]> data() throws Exception {
            return standardDatasets;
        }
    }
}
