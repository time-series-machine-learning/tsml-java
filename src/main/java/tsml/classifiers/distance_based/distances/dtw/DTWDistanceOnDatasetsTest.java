package tsml.classifiers.distance_based.distances.dtw;

import org.junit.runners.Parameterized;
import tsml.classifiers.distance_based.distances.DistanceMeasureOnDatasetsTest;
import tsml.classifiers.distance_based.distances.DistanceMeasureSpaceBuilder;

import java.util.Collection;

public class DTWDistanceOnDatasetsTest
        extends DistanceMeasureOnDatasetsTest {

    @Override public DistanceMeasureSpaceBuilder getBuilder() {
        return DistanceMeasureSpaceBuilder.DTW;
    }

    @Parameterized.Parameters(name = "{0}")
    public static Collection<Object[]> data() throws Exception {
        return standardDatasets;
    }
}
