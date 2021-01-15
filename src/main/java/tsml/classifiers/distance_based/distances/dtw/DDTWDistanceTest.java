package tsml.classifiers.distance_based.distances.dtw;

import org.junit.runners.Parameterized;
import tsml.classifiers.distance_based.distances.DistanceMeasureOnDatasetsTest;
import tsml.classifiers.distance_based.distances.dtw.spaces.DDTWDistanceSpace;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpaceBuilder;

import java.util.Collection;

public class DDTWDistanceTest {


    public static class DistanceMeasureDatasetsTest
            extends DistanceMeasureOnDatasetsTest {

        @Override public ParamSpaceBuilder getBuilder() {
            return new DDTWDistanceSpace();
        }

        @Parameterized.Parameters(name = "{0}")
        public static Collection<Object[]> data() throws Exception {
            return standardDatasets;
        }
    }
}
