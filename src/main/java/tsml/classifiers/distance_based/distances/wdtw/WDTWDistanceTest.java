package tsml.classifiers.distance_based.distances.wdtw;

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

public class WDTWDistanceTest {


    private static DistanceTester buildDistanceFinder() {
        return new DistanceTester() {
            private ParamSpace space;
            private Instances data;

            @Override
            public void findDistance(final Random random, final Instances data, final Instance ai,
                final Instance bi, final double limit) {
                if(data != this.data) {
                    this.data = data;
                    space = WDTWDistanceConfigs.buildWDTWParams();
                }
                final GridSearchIterator iterator = new GridSearchIterator(space);
//                                                int i = 0;
                while(iterator.hasNext()) {
//                                                            System.out.println("i:" + i++);
                    final ParamSet paramSet = iterator.next();
                    final double g = (double) paramSet.get(WDTW.G_FLAG).get(0);
                    // doesn't test window, MSM originally doesn't have window
                    //                    final int window = (int) paramSet.get(MSMDistance.).get(0);
                    final WDTWDistance df = new WDTWDistance();
                    df.setG(g);
//                    df.setKeepMatrix(true);
                    Assert.assertEquals(df.distance(ai, bi, limit), origWdtw(ai, bi, limit, g), 0);
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

    private static double[] generateWeights(int seriesLength, double g) {
        double halfLength = (double) seriesLength / 2;
        double[] weightVector = new double[seriesLength];
        for (int i = 0; i < seriesLength; i++) {
            weightVector[i] = 1d / (1d + Math.exp(-g * (i - halfLength)));
        }
        return weightVector;
    }

    private static double origWdtw(Instance a, Instance b, double limit, double g) {

        int aLength = a.numAttributes() - 1;
        int bLength = b.numAttributes() - 1;

        double[] weightVector = generateWeights(aLength, g);

        //create empty array
        double[][] distances = new double[aLength][bLength];

        //first value
        distances[0][0] = (a.value(0) - b.value(0)) * (a.value(0) - b.value(0)) * weightVector[0];

        //early abandon if first values is larger than cut off
        if (distances[0][0] > limit) {
            return Double.POSITIVE_INFINITY;
        }

        //top row
        for (int i = 1; i < bLength; i++) {
            distances[0][i] =
                distances[0][i - 1] + (a.value(0) - b.value(i)) * (a.value(0) - b.value(i)) * weightVector[i]; //edited by Jay
        }

        //first column
        for (int i = 1; i < aLength; i++) {
            distances[i][0] =
                distances[i - 1][0] + (a.value(i) - b.value(0)) * (a.value(i) - b.value(0)) * weightVector[i]; //edited by Jay
        }

        //warp rest
        double minDistance;
        for (int i = 1; i < aLength; i++) {
            boolean overflow = true;

            for (int j = 1; j < bLength; j++) {
                //calculate distance_measures
                minDistance = Math.min(distances[i][j - 1], Math.min(distances[i - 1][j], distances[i - 1][j - 1]));
                distances[i][j] =
                    minDistance + (a.value(i) - b.value(j)) * (a.value(i) - b.value(j)) * weightVector[Math.abs(i - j)];

                if (overflow && distances[i][j] <= limit) {
                    overflow = false; // because there's evidence that the path can continue
                }
            }

            //early abandon
            if (overflow) {
                return Double.POSITIVE_INFINITY;
            }
        }
        return distances[aLength - 1][bLength - 1];
    }
}
