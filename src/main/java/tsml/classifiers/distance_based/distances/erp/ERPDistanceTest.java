package tsml.classifiers.distance_based.distances.erp;

import java.util.Random;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import tsml.classifiers.distance_based.distances.dtw.DTWDistanceTest;
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

    @Test
    public void testRandomSetups() {
        final int min = -5;
        final int max = 5;
        final int length = 100;
        final int count = 10000;
        final double[] resultsOrig = new double[count];
        final double[] resultsCustom = new double[count];
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
            final double penalty = Math.abs(max - min) * random.nextDouble();
            int window;
            if(random.nextBoolean()) {
                window = random.nextInt(100);
            } else {
                window = -1;
            }
            double limit = Double.POSITIVE_INFINITY;
            if(random.nextBoolean()) {
                limit = random.nextDouble() * 1000;
            }
            resultsOrig[i] = origErp(ai, bi, limit, window, penalty);
            if(resultsOrig[i] == Double.POSITIVE_INFINITY) {
                limitCount++;
            }
            final ERPDistance df = new ERPDistance();
            df.setPenalty(penalty);
            df.setWindowSize(window);
            this.df.setKeepMatrix(true);
            resultsCustom[i] = df.distance(ai, bi, limit);
        }
        Assert.assertArrayEquals(resultsCustom,  resultsOrig, 0d);
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
