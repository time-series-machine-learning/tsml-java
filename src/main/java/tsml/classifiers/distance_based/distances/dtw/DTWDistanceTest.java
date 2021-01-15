package tsml.classifiers.distance_based.distances.dtw;

import java.io.File;
import java.util.*;
import java.util.concurrent.TimeUnit;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import tsml.data_containers.utilities.Converter;
import tsml.transformers.Derivative;
import utilities.FileUtils;
import utilities.InstanceTools;
import weka.core.Instance;
import weka.core.Instances;

import static experiments.data.DatasetLoading.*;

/**
 * Purpose: test dtw
 * <p>
 * Contributors: goastler
 */
public class DTWDistanceTest {
    
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
        df.buildDistanceMeasure(instances);
    }

    @Test
    public void testFullWarp() {
        df.setWindowSize(1);
        double distance = df.distance(instances.get(0), instances.get(1));
        Assert.assertEquals(distance, 203, 0);
    }

    @Test
    public void testConstrainedWarp() {
        df.setWindowSize(0.4);
        double distance = df.distance(instances.get(0), instances.get(1));
        Assert.assertEquals(distance, 212, 0);
    }

    @Test
    public void testVariableLengthTimeSeries() {
        DTWDistance dtw = new DTWDistance();
        dtw.setGenerateDistanceMatrix(true);
        dtw.setWindowSize(1);
        TimeSeriesInstances tsinsts = new TimeSeriesInstances(new double[][][]{
                {
                        {7, 6, 1, 7, 7, 7, 3, 3, 5, 6}
                },
                {
                        {5, 3, 2, 7, 4, 2, 1, 8, 8, 7, 4, 4, 2, 1, 3}
                }
        }, new double[]{0, 0});
        double distance = dtw.distance(tsinsts.get(0), tsinsts.get(1));
        Assert.assertEquals(57, distance, 0d);
//        System.out.println("[" + ArrayUtilities.toString(dtw.getDistanceMatrix(), ",", "]," + System.lineSeparator() + "[") + "]");
        double otherDistance = dtw.distance(tsinsts.get(1), tsinsts.get(0));
        Assert.assertEquals(distance, otherDistance, 0d);
        double limit = 10;
        distance = dtw.distance(tsinsts.get(0), tsinsts.get(1), limit);
        Assert.assertEquals(Double.POSITIVE_INFINITY, distance, 0d);
        otherDistance = dtw.distance(tsinsts.get(1), tsinsts.get(0), limit);
        Assert.assertEquals(distance, otherDistance, 0d);
    }
    
    @Test
    public void testVariableLengthTimeSeriesConstrainedWarp() {
        DTWDistance dtw = new DTWDistance();
        dtw.setGenerateDistanceMatrix(true);
        dtw.setWindowSize(0.25);
        TimeSeriesInstances tsinsts = new TimeSeriesInstances(new double[][][]{
                {
                        {7, 6, 1, 7, 7, 7, 3, 3, 5, 6}
                },
                {
                        {5, 3, 2, 7, 4, 2, 1, 8, 8, 7, 4, 4, 2, 1, 3}
                }
        }, new double[]{0, 0});
        double distance = dtw.distance(tsinsts.get(0), tsinsts.get(1));
        Assert.assertEquals(57, distance, 0d);
        //        System.out.println("[" + ArrayUtilities.toString(dtw.getDistanceMatrix(), ",", "]," + System.lineSeparator() + "[") + "]");
        double otherDistance = dtw.distance(tsinsts.get(1), tsinsts.get(0));
        Assert.assertEquals(distance, otherDistance, 0d);
        double limit = 20;
        distance = dtw.distance(tsinsts.get(0), tsinsts.get(1), limit);
        Assert.assertEquals(Double.POSITIVE_INFINITY, distance, 0d);
        otherDistance = dtw.distance(tsinsts.get(1), tsinsts.get(0), limit);
        Assert.assertEquals(distance, otherDistance, 0d);
    }
}
