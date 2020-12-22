package tsml.classifiers.distance_based.distances.wdtw;

import java.util.Random;
import org.junit.Assert;
import org.junit.Test;
import tsml.classifiers.distance_based.distances.DistanceMeasureTest;
import tsml.classifiers.distance_based.distances.erp.ERPDistanceTest;
import tsml.classifiers.distance_based.distances.erp.ERPDistanceTest.DistanceTester;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpace;
import tsml.classifiers.distance_based.utils.collections.params.iteration.RandomSearch;
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
                final RandomSearch iterator = new RandomSearch();
                iterator.setRandom(random);
                iterator.buildSearch(space);
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
                    Assert.assertEquals(df.distance(ai, bi, limit), DistanceMeasureTest.origWdtw(ai, bi, limit, g), 0);
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


}
