package tsml.classifiers.distance_based.distances.lcss;

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

public class LCSSDistanceTest {

    private static DistanceTester buildDistanceFinder() {
        return new DistanceTester() {
            private ParamSpace space;
            private Instances data;

            @Override
            public void findDistance(final Random random, final Instances data, final Instance ai,
                final Instance bi, final double limit) {
                if(data != this.data) {
                    this.data = data;
                    space = LCSSDistanceConfigs.buildLCSSParams(data);
                }
                final GridSearchIterator iterator = new GridSearchIterator(space);
//                int i = 0;
                while(iterator.hasNext()) {
//                    System.out.println("i:" + i++);
                    final ParamSet paramSet = iterator.next();
                    final double epsilon = (double) paramSet.get(LCSSDistance.EPSILON_FLAG).get(0);
                    final int window = (int) paramSet.get(LCSSDistance.WINDOW_SIZE_FLAG).get(0);
                    final LCSSDistance df = new LCSSDistance();
                    df.setEpsilon(epsilon);
                    df.setWindowSize(window);
                    Assert.assertEquals(df.distance(ai, bi, limit), origLcss(ai, bi, limit, window, epsilon), 0);
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

    private static double origLcss(Instance a, Instance b, double limit, int delta, double epsilon) {

        int aLength = a.numAttributes() - 1;
        int bLength = b.numAttributes() - 1;

        // 22/10/19 goastler - limit LCSS such that if any value in the current window is larger than the limit then we can stop here, no point in doing the extra work
        if(limit != Double.POSITIVE_INFINITY) { // check if there's a limit set
            // if so then reverse engineer the max LCSS distance and replace the limit
            // this is just the inverse of the return value integer rounded to an LCSS distance
            limit = (int) ((1 - limit) * aLength) + 1;
        }

        int[][] lcss = new int[aLength + 1][bLength + 1];

        int warpingWindow = delta;
        if(warpingWindow < 0) {
            warpingWindow = aLength + 1;
        }

        for(int i = 0; i < aLength; i++) {
            boolean tooBig = true;
            for(int j = i - warpingWindow; j <= i + warpingWindow; j++) {
                if(j < 0) {
                    j = -1;
                } else if(j >= bLength) {
                    j = i + warpingWindow;
                } else {
                    if(b.value(j) + epsilon >= a.value(i) && b.value(j) - epsilon <= a
                        .value(i)) {
                        lcss[i + 1][j + 1] = lcss[i][j] + 1;
                        //                    } else if(lcss[i][j + 1] > lcss[i + 1][j]) {
                        //                        lcss[i + 1][j + 1] = lcss[i][j + 1];
                        //                    } else {
                        //                        lcss[i + 1][j + 1] = lcss[i + 1][j];
                    }
                    else {
                        lcss[i + 1][j + 1] = Math.max(lcss[i + 1][j], Math.max(lcss[i][j], lcss[i][j + 1]));
                    }
                    // if this value is less than the limit then fast-fail the limit overflow
                    if(tooBig && lcss[i + 1][j + 1] <= limit) {
                        tooBig = false;
                    }
                }
            }

            // if no element is lower than the limit then early abandon
            if(tooBig) {
                return Double.POSITIVE_INFINITY;
            }

        }

        int max = -1;
        for(int j = 1; j < lcss[lcss.length - 1].length; j++) {
            if(lcss[lcss.length - 1][j] > max) {
                max = lcss[lcss.length - 1][j];
            }
        }
        return 1 - ((double) max / aLength);
    }
}
