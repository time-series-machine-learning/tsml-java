package tsml.classifiers.distance_based.distances.erp;

import experiments.data.DatasetLoading;

import java.util.Collection;
import java.util.Random;
import java.util.concurrent.TimeUnit;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.junit.runners.Parameterized;
import tsml.classifiers.distance_based.distances.DistanceMeasureSpaceBuilder;
import tsml.classifiers.distance_based.distances.dtw.DTW;
import tsml.classifiers.distance_based.distances.dtw.DTWDistanceTest;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpace;
import tsml.classifiers.distance_based.utils.collections.params.iteration.RandomSearch;
import utilities.InstanceTools;
import weka.core.Instance;
import weka.core.Instances;

public class ERPDistanceTest {
    private Instances instances;
    private ERPDistance df;

    @Before
    public void before() {
        instances = DTWDistanceTest.buildInstances();
        df = new ERPDistance();
        df.buildDistanceMeasure(instances);
    }

    @Test
    public void testFullWarpA() {
        df.setWindowSize(1);
        df.setG(1.5);
        double distance = df.distance(instances.get(0), instances.get(1));
        Assert.assertEquals(distance, 182, 0);
    }

    @Test
    public void testFullWarpB() {
        df.setWindowSize(1);
        df.setG(2);
        double distance = df.distance(instances.get(0), instances.get(1));
        Assert.assertEquals(distance, 175, 0);
    }

    @Test
    public void testConstrainedWarpA() {
        df.setWindowSize(0.2);
        df.setG(1.5);
        double distance = df.distance(instances.get(0), instances.get(1));
        Assert.assertEquals(distance, 189.5, 0);
    }

    @Test
    public void testConstrainedWarpB() {
        df.setWindowSize(1);
        df.setG(2);
        double distance = df.distance(instances.get(0), instances.get(1));
        Assert.assertEquals(distance, 175, 0);
    }
    
    public static class TestOnDatasets extends DTWDistanceTest.TestOnDatasets {

        @Override public DistanceMeasureSpaceBuilder getBuilder() {
            return DistanceMeasureSpaceBuilder.ERP;
        }

        @Parameterized.Parameters(name = "{0}")
        public static Collection<Object[]> data() throws Exception {
            return standardDatasets;
        }
    }

}
