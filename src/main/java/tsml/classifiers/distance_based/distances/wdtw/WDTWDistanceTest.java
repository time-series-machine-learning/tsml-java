package tsml.classifiers.distance_based.distances.wdtw;

import java.util.Collection;
import java.util.Random;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runners.Parameterized;
import tsml.classifiers.distance_based.distances.DistanceMeasureSpaceBuilder;
import tsml.classifiers.distance_based.distances.dtw.DTWDistanceTest;
import tsml.classifiers.distance_based.distances.erp.ERPDistanceTest;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpace;
import tsml.classifiers.distance_based.utils.collections.params.iteration.RandomSearch;
import weka.core.Instance;
import weka.core.Instances;

public class WDTWDistanceTest {



    public static class TestOnDatasets extends DTWDistanceTest.TestOnDatasets {

        @Override public DistanceMeasureSpaceBuilder getBuilder() {
            return DistanceMeasureSpaceBuilder.ED;
        }

        @Parameterized.Parameters(name = "{0}")
        public static Collection<Object[]> data() throws Exception {
            return standardDatasets;
        }
    }

}
