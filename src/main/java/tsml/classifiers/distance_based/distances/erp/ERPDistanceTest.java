package tsml.classifiers.distance_based.distances.erp;

import distance.elastic.DistanceMeasure;
import experiments.data.DatasetLoading;
import java.util.List;
import java.util.Random;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import tsml.classifiers.distance_based.distances.DistanceMeasureable;
import tsml.classifiers.distance_based.distances.dtw.DTWDistanceTest;
import tsml.classifiers.distance_based.utils.params.ParamSet;
import tsml.classifiers.distance_based.utils.params.ParamSpace;
import tsml.classifiers.distance_based.utils.params.iteration.GridSearchIterator;
import utilities.InstanceTools;
import weka.core.DistanceFunction;
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
        df.setPenalty(1.5);
        double distance = df.distance(instances.get(0), instances.get(1));
        Assert.assertEquals(distance, 182, 0);
    }

    @Test
    public void testFullWarpB() {
        df.setWindowSize(-1);
        df.setPenalty(2);
        double distance = df.distance(instances.get(0), instances.get(1));
        Assert.assertEquals(distance, 175, 0);
    }

    @Test
    public void testConstrainedWarpA() {
        df.setWindowSize(1);
        df.setPenalty(1.5);
        double distance = df.distance(instances.get(0), instances.get(1));
        Assert.assertEquals(distance, 189.5, 0);
    }

    @Test
    public void testConstrainedWarpB() {
        df.setWindowSize(1);
        df.setPenalty(2);
        double distance = df.distance(instances.get(0), instances.get(1));
        Assert.assertEquals(distance, 189, 0);
    }

    public interface DistanceFinder {
        double[] findDistance(Random random, int min, int max, int length, Instance ai, Instance bi, double limit);
    }

    public static void runGunPointDistanceFunctionTests(ParamSpace space) throws Exception {
        final Instances[] instances = DatasetLoading.sampleGunPoint(0);
        final Instances data = instances[0];
        data.addAll(instances[1]);
        for(int i = 1; i < data.size(); i++) {
            final Instance a = data.get(i);
            for(int j = 0; j < i; j++) {
                final Instance b = data.get(j);
                final GridSearchIterator iterator = new GridSearchIterator(space);
                while(iterator.hasNext()) {
                    final ParamSet paramSet = iterator.next();
                    final List<Object> list = paramSet.get(DistanceMeasureable.DISTANCE_MEASURE_FLAG);
                    Assert.assertEquals(list.size(), 1);
                    final DistanceFunction df = (DistanceFunction) list.get(0);
                    final double distance = df.distance(a, b);
                }
            }
        }
    }

    public static void runRandomDistanceFunctionTests(DistanceFinder df) {
        final int min = -5;
        final int max = 5;
        final int length = 100;
        final int count = 10000;
        int limitCount = 0;
        for(int i = 0; i < count; i++) {
            final Random random = new Random(i);
            final double[] a = buildRandomArray(random, length, min, max);
            final double[] b = buildRandomArray(random, length, min, max);
            a[a.length - 1] = 1;
            b[b.length - 1] = 1;
            final Instances instances = InstanceTools.toWekaInstances(new double[][]{a, b}, new double[] {1,1});
            final Instance ai = instances.get(0);
            final Instance bi = instances.get(1);
            double limit = Double.POSITIVE_INFINITY;
            if(random.nextBoolean()) {
                limit = random.nextDouble() * 1000;
            }
            final double[] distances = df.findDistance(random, min, max, length, ai, bi, limit);
            if(distances[0] == Double.POSITIVE_INFINITY) {
                limitCount++;
            }
            for(int j = 1; j < distances.length; j++) {
                Assert.assertEquals(distances[0], distances[j], 0);
            }
        }
    }

    @Test
    public void testRandomSetups() {
        runRandomDistanceFunctionTests((random, min, max, length, ai, bi, limit) -> {
            final double penalty = Math.abs(max - min) * random.nextDouble();
            int window;
            if(random.nextBoolean()) {
                window = random.nextInt(length);
            } else {
                window = -1;
            }
            final ERPDistance df = new ERPDistance();
            df.setKeepMatrix(true);
            df.setWindowSize(window);
            df.setPenalty(penalty);
            return new double[] {df.distance(ai, bi, limit), origErp(ai, bi, limit, window, penalty)};
        });
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
