package tsml.classifiers.distance_based.distances.dtw;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import tsml.classifiers.distance_based.utils.collections.params.ParamHandlerTest;
import tsml.data_containers.TimeSeriesInstances;
import utilities.InstanceTools;
import weka.core.Instances;

import static tsml.classifiers.distance_based.distances.dtw.spaces.DDTWDistanceSpace.newDDTWDistance;

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
        df.setWindow(1);
        double distance = df.distance(instances.get(0), instances.get(1));
        Assert.assertEquals(distance, 203, 0);
    }

    @Test
    public void testConstrainedWarp() {
        df.setWindow(0.4);
        double distance = df.distance(instances.get(0), instances.get(1));
        Assert.assertEquals(distance, 212, 0);
    }

    @Test
    public void testVariableLengthTimeSeries() {
        DTWDistance dtw = new DTWDistance();
        dtw.setRecordCostMatrix(true);
        dtw.setWindow(1);
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
        dtw.setRecordCostMatrix(true);
        dtw.setWindow(0.25);
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
    
    public static class DTWParamTest extends ParamHandlerTest {

        @Override public Object getHandler() {
            return new DTWDistance();
        }
    }

    public static class DDTWParamTest extends ParamHandlerTest {

        @Override public Object getHandler() {
            return newDDTWDistance();
        }
    }
}
