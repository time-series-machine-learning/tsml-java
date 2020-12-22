package tsml.classifiers.distance_based.distances.erp;

import experiments.data.DatasetLoading;
import java.util.Random;
import java.util.concurrent.TimeUnit;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import tsml.classifiers.distance_based.distances.DistanceMeasureTest;
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

    public interface DistanceTester {
        void findDistance(Random random, Instances data, Instance ai, Instance bi, double limit);
    }

    public static void testDistanceFunctionsOnGunPoint(DistanceTester df) throws Exception {
        testDistanceFunctionOnDataset(DatasetLoading.loadGunPoint(), df);
    }

    public static void testDistanceFunctionsOnBeef(DistanceTester df) throws Exception {
        testDistanceFunctionOnDataset(DatasetLoading.loadBeef(), df);
    }

    public static void testDistanceFunctionsOnItalyPowerDemand(DistanceTester df) throws Exception {
        testDistanceFunctionOnDataset(DatasetLoading.loadItalyPowerDemand(), df);
    }

    public static void testDistanceFunctionsOnRandomDataset(DistanceTester df) {
        testDistanceFunctionOnDataset(buildRandomDataset(new Random(0), -5, 5, 100, 100, 2), df);
    }

    public static void testDistanceFunctionOnDataset(Instances data, DistanceTester df) {
        Random random = new Random(0);
        final int score = data.size() * data.numAttributes();
        int instanceCount = 0;
        int attributeCount = 0;
        final long timeStamp = System.nanoTime();
        final long timeLimit = TimeUnit.NANOSECONDS.convert(10000, TimeUnit.MILLISECONDS);
        for(int i = 0; i < data.size(); i++) {
//            final Instance a = data.get(i);
            final Instance a = data.get(random.nextInt(data.size()));
            for(int j = 0; j < i; j++) {
                instanceCount++;
                attributeCount += Math.pow(data.numAttributes(), 2);
                //                final Instance b = data.get(j);
                final Instance b = data.get(random.nextInt(data.size()));
                double limit = random.nextDouble() * 2 * (data.numAttributes() - 1);
                df.findDistance(random, data, a, b, limit);
                limit = Double.POSITIVE_INFINITY;
                df.findDistance(random, data, a, b, limit);
                // quick exit for speedy unit testing. Turn this off to do full blown testing (takes ~1hr)
                if(System.nanoTime() - timeStamp > timeLimit && instanceCount >= 10) {
//                    System.out.println(instanceCount);
                    return;
                }
            }
        }
    }

    public static Instances buildRandomDataset(Random random, double min, double max, int length, int count, int numClasses) {
        double[][] data = new double[count][];
        double[] labels = new double[count];
        for(int i = 0; i < count; i++) {
            final double[] a = buildRandomArray(random, length, min, max);
            labels[i] = random.nextInt(numClasses);
            data[i] = a;
        }
        return InstanceTools.toWekaInstances(data, labels);
    }

    private static DistanceTester buildDistanceFinder() {
        return new DistanceTester() {
            private ParamSpace space;
            private Instances data;

            @Override
            public void findDistance(final Random random, final Instances data, final Instance ai,
                final Instance bi, final double limit) {
                if(data != this.data) {
                    this.data = data;
                    space = ERPDistanceConfigs.buildERPParams(data);
                }
                final RandomSearch iterator = new RandomSearch();
                iterator.setRandom(random);
                iterator.buildSearch(space);
//                int i = 0;
                while(iterator.hasNext()) {
//                    System.out.println("i:" + i++);
                    final ParamSet paramSet = iterator.next();
                    final double penalty = (double) paramSet.get(ERPDistance.G_FLAG).get(0);
                    final double window = (double) paramSet.get(DTW.WINDOW_SIZE_FLAG).get(0);
                    final int len = ai.numAttributes() - 1;
                    final int rawWindow = (int) Math.floor(window * len);
                    final double otherWindow = Math.min(1, Math.max(0, ((double) rawWindow) / len));
                    //                    System.out.println(window);
                    //                    System.out.println(rawWindow + " " + (otherWindow * (len - 1)));
                    final ERPDistance df = new ERPDistance();
                    df.setWindowSize(otherWindow);
                    df.setG(penalty);
                    df.setGenerateDistanceMatrix(false);
                    final double a = df.distance(ai, bi, limit);
                    final double b = DistanceMeasureTest.origErp(ai, bi, limit, rawWindow, penalty);
                    df.setGenerateDistanceMatrix(true);
                    final double c = df.distance(ai, bi, limit);
                    Assert.assertEquals(a, b,  0);
                    Assert.assertEquals(a, c,  0);
                }
            }
        };
    }

    @Test
    public void testBeef() throws Exception {
        testDistanceFunctionsOnBeef(buildDistanceFinder());
    }

    @Test
    public void testGunPoint() throws Exception {
        testDistanceFunctionsOnGunPoint(buildDistanceFinder());
    }

    @Test
    public void testItalyPowerDemand() throws Exception {
        testDistanceFunctionsOnItalyPowerDemand(buildDistanceFinder());
    }

    @Test
    public void testRandomDataset() throws Exception {
        testDistanceFunctionsOnRandomDataset(buildDistanceFinder());
    }

    public static double[] buildRandomArray(Random random, int length, double min, double max) {
        double diff = Math.abs(min - max);
        min = Math.min(min, max);
        double[] array = new double[length];
        for(int i = 0; i < length; i++) {
            array[i] = random.nextDouble() * diff + min;
        }
        return array;
    }

}
