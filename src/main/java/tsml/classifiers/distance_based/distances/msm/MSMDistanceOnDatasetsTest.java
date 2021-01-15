package tsml.classifiers.distance_based.distances.msm;

import org.junit.runners.Parameterized;
import tsml.classifiers.distance_based.distances.DistanceMeasureOnDatasetsTest;
import tsml.classifiers.distance_based.distances.msm.spaces.MSMDistanceSpace;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpaceBuilder;

import java.util.Collection;

public class MSMDistanceOnDatasetsTest
        extends DistanceMeasureOnDatasetsTest {

    @Override public ParamSpaceBuilder getBuilder() {
        return new MSMDistanceSpace();
    }

    @Parameterized.Parameters(name = "{0}")
    public static Collection<Object[]> data() throws Exception {
        return standardDatasets;
    }
}
