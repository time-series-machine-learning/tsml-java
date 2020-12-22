package tsml.classifiers.distance_based.distances.dtw;

import org.junit.runners.Parameterized;
import tsml.classifiers.distance_based.distances.DistanceMeasureSpaceBuilder;

import java.util.Collection;

import static tsml.classifiers.distance_based.distances.dtw.DTWDistanceTest.*;

public class DDTWDistanceTest {


    public static class TestOnDatasets extends DTWDistanceTest.TestOnDatasets {

        @Override public DistanceMeasureSpaceBuilder getBuilder() {
            return DistanceMeasureSpaceBuilder.DDTW;
        }

        @Parameterized.Parameters(name = "{0}")
        public static Collection<Object[]> data() throws Exception {
            return standardDatasets;
        }
    }
}
