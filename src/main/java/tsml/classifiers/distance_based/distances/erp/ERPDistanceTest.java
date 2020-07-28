package tsml.classifiers.distance_based.distances.erp;

import experiments.data.DatasetLoading;
import java.util.Random;
import java.util.concurrent.TimeUnit;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import tsml.classifiers.distance_based.distances.dtw.DTWDistanceTest;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpace;
import tsml.classifiers.distance_based.utils.collections.params.iteration.GridSearchIterator;
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
        df.setInstances(instances);
    }

    @Test
    public void testFullWarpA() {
        df.setWindowSize(-1);
        df.setG(1.5);
        double distance = df.distance(instances.get(0), instances.get(1));
        Assert.assertEquals(distance, 182, 0);
    }

    @Test
    public void testFullWarpB() {
        df.setWindowSize(-1);
        df.setG(2);
        double distance = df.distance(instances.get(0), instances.get(1));
        Assert.assertEquals(distance, 175, 0);
    }

    @Test
    public void testConstrainedWarpA() {
        df.setWindowSize(1);
        df.setG(1.5);
        double distance = df.distance(instances.get(0), instances.get(1));
        Assert.assertEquals(distance, 189.5, 0);
    }

    @Test
    public void testConstrainedWarpB() {
        df.setWindowSize(1);
        df.setG(2);
        double distance = df.distance(instances.get(0), instances.get(1));
        Assert.assertEquals(distance, 189, 0);
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
        final long timeLimit = TimeUnit.NANOSECONDS.convert(5000, TimeUnit.MILLISECONDS);
        for(int i = 0; i < data.size(); i++) {
//            final Instance a = data.get(i);
            final Instance a = data.get(random.nextInt(data.size()));
            for(int j = 0; j < i; j++) {
                instanceCount++;
                attributeCount += Math.pow(data.numAttributes(), 2);
                //                final Instance b = data.get(j);
                final Instance b = data.get(random.nextInt(data.size()));
                double limit = random.nextDouble() * 2 * data.numAttributes() - 1;
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
                final GridSearchIterator iterator = new GridSearchIterator(space);
//                int i = 0;
                while(iterator.hasNext()) {
//                    System.out.println("i:" + i++);
                    final ParamSet paramSet = iterator.next();
                    final double penalty = (double) paramSet.get(ERPDistance.G_FLAG).get(0);
                    final int window = (int) paramSet.get(ERPDistance.WINDOW_SIZE_FLAG).get(0);
                    final ERPDistance df = new ERPDistance();
                    df.setWindowSize(window);
                    df.setG(penalty);
                    Assert.assertEquals(df.distance(ai, bi, limit), origErp(ai, bi, limit, window, penalty), 0);
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

    private static double origErp(Instance first, Instance second, double limit, int band, double penalty) {

        int aLength = first.numAttributes() - 1;
        int bLength = second.numAttributes() - 1;

        // Current and previous columns of the matrix
        double[] curr = new double[bLength];
        double[] prev = new double[bLength];

        // size of edit distance band
        // bandsize is the maximum allowed distance to the diagonal
        //        int band = (int) Math.ceil(v2.getDimensionality() * bandSize);
        if(band < 0) {
            band = aLength + 1;
        }

        // g parameters for local usage
        double gValue = penalty;

        for(int i = 0;
            i < aLength;
            i++) {
            // Swap current and prev arrays. We'll just overwrite the new curr.
            {
                double[] temp = prev;
                prev = curr;
                curr = temp;
            }
            int l = i - (band + 1);
            if(l < 0) {
                l = 0;
            }
            int r = i + (band + 1);
            if(r > (bLength - 1)) {
                r = (bLength - 1);
            }

            boolean tooBig = true;

            for(int j = l;
                j <= r;
                j++) {
                if(Math.abs(i - j) <= band) {
                    // compute squared distance of feature vectors
                    double val1 = first.value(i);
                    double val2 = gValue;
                    double diff = (val1 - val2);
                    final double dist1 = diff * diff;

                    val1 = gValue;
                    val2 = second.value(j);
                    diff = (val1 - val2);
                    final double dist2 = diff * diff;

                    val1 = first.value(i);
                    val2 = second.value(j);
                    diff = (val1 - val2);
                    final double dist12 = diff * diff;

                    final double cost;

                    if((i + j) != 0) {
                        if((i == 0) || ((j != 0) && (((prev[j - 1] + dist12) > (curr[j - 1] + dist2)) && (
                            (curr[j - 1] + dist2) < (prev[j] + dist1))))) {
                            // del
                            cost = curr[j - 1] + dist2;
                        } else if((j == 0) || ((i != 0) && (((prev[j - 1] + dist12) > (prev[j] + dist1)) && (
                            (prev[j] + dist1) < (curr[j - 1] + dist2))))) {
                            // ins
                            cost = prev[j] + dist1;
                        } else {
                            // match
                            cost = prev[j - 1] + dist12;
                        }
                    } else {
                        cost = 0;
                    }

                    curr[j] = cost;

                    if(tooBig && cost < limit) {
                        tooBig = false;
                    }
                } else {
                    curr[j] = Double.POSITIVE_INFINITY; // outside band
                }
            }
            if(tooBig) {
                return Double.POSITIVE_INFINITY;
            }
        }

        return curr[bLength - 1];
    }
}
