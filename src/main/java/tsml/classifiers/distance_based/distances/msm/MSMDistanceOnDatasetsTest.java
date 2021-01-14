package tsml.classifiers.distance_based.distances.msm;

import org.junit.runners.Parameterized;
import tsml.classifiers.distance_based.distances.DistanceMeasureOnDatasetsTest;
import tsml.classifiers.distance_based.distances.DistanceMeasureSpaceBuilder;

import java.util.Collection;

public class MSMDistanceOnDatasetsTest
        extends DistanceMeasureOnDatasetsTest {

    @Override public DistanceMeasureSpaceBuilder getBuilder() {
        return DistanceMeasureSpaceBuilder.MSM;
    }

    @Parameterized.Parameters(name = "{0}")
    public static Collection<Object[]> data() throws Exception {
        return standardDatasets;
    }
}
