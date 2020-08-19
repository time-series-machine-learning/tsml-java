package tsml.classifiers.distance_based.distances.msm;

import java.util.Random;
import org.junit.Assert;
import org.junit.Test;
import tsml.classifiers.distance_based.distances.erp.ERPDistanceTest;
import tsml.classifiers.distance_based.distances.erp.ERPDistanceTest.DistanceTester;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpace;
import tsml.classifiers.distance_based.utils.collections.params.iteration.GridSearchIterator;
import weka.core.Instance;
import weka.core.Instances;

public class MSMDistanceTest {


    private static DistanceTester buildDistanceFinder() {
        return new DistanceTester() {
            private ParamSpace space;
            private Instances data;

            @Override
            public void findDistance(final Random random, final Instances data, final Instance ai,
                final Instance bi, final double limit) {
                if(data != this.data) {
                    this.data = data;
                    space = MSMDistanceConfigs.buildMSMParams();
                }
                final GridSearchIterator iterator = new GridSearchIterator(space);
//                                int i = 0;
                while(iterator.hasNext()) {
//                                        System.out.println("i:" + i++);
                    final ParamSet paramSet = iterator.next();
                    final double cost = (double) paramSet.get(MSMDistance.C_FLAG).get(0);
                    // doesn't test window, MSM originally doesn't have window
//                    final int window = (int) paramSet.get(MSMDistance.).get(0);
                    final MSMDistance df = new MSMDistance();
                    df.setC(cost);
                    Assert.assertEquals(df.distance(ai, bi, limit), origMsm(ai, bi, limit, cost), 0);
                }
            }
        };
    }

    @Test
    public void testBeef() throws Exception {
        ERPDistanceTest.testDistanceFunctionsOnBeef(buildDistanceFinder());
    }

    @Test
    public void testGunPoint() throws Exception {
        ERPDistanceTest.testDistanceFunctionsOnGunPoint(buildDistanceFinder());
    }

    @Test
    public void testItalyPowerDemand() throws Exception {
        ERPDistanceTest.testDistanceFunctionsOnItalyPowerDemand(buildDistanceFinder());
    }

    @Test
    public void testRandomDataset() {
        ERPDistanceTest.testDistanceFunctionsOnRandomDataset(buildDistanceFinder());
    }

    private static double findCost(double newPoint, double x, double y, double c) {
        double dist = 0;

        if(((x <= newPoint) && (newPoint <= y)) ||
            ((y <= newPoint) && (newPoint <= x))) {
            dist = c;
        } else {
            dist = c + Math.min(Math.abs(newPoint - x), Math.abs(newPoint - y));
        }

        return dist;
    }

    private static double origMsm(Instance a, Instance b, double limit, double c) {

        int aLength = a.numAttributes() - 1;
        int bLength = b.numAttributes() - 1;

        double[][] cost = new double[aLength][bLength];

        // Initialization
        cost[0][0] = Math.abs(a.value(0) - b.value(0));
        for(int i = 1; i < aLength; i++) {
            cost[i][0] = cost[i - 1][0] + findCost(a.value(i), a.value(i - 1), b.value(0), c);
        }
        for(int i = 1; i < bLength; i++) {
            cost[0][i] = cost[0][i - 1] + findCost(b.value(i), a.value(0), b.value(i - 1), c);
        }

        // Main Loop
        double min;
        for(int i = 1; i < aLength; i++) {
            min = Double.POSITIVE_INFINITY;
            for(int j = 1; j < bLength; j++) {
                double d1, d2, d3;
                d1 = cost[i - 1][j - 1] + Math.abs(a.value(i) - b.value(j));
                d2 = cost[i - 1][j] + findCost(a.value(i), a.value(i - 1), b.value(j), c);
                d3 = cost[i][j - 1] + findCost(b.value(j), a.value(i), b.value(j - 1), c);
                cost[i][j] = Math.min(d1, Math.min(d2, d3));

            }
            for(int j = 0; j < bLength; j++) {
                min = Math.min(min, cost[i][j]);
            }
            if(min > limit) {
                return Double.POSITIVE_INFINITY;
            }
        }
        // Output
        return cost[aLength - 1][bLength - 1];
    }
}
