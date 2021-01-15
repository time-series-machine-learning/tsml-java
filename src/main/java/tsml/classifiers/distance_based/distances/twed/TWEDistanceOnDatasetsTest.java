package tsml.classifiers.distance_based.distances.twed;

import org.junit.runners.Parameterized;
import tsml.classifiers.distance_based.distances.DistanceMeasureOnDatasetsTest;
import tsml.classifiers.distance_based.distances.twed.spaces.TWEDistanceSpace;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpaceBuilder;

import java.util.Collection;

public class TWEDistanceOnDatasetsTest
        extends DistanceMeasureOnDatasetsTest {

    @Override public ParamSpaceBuilder getBuilder() {
        return new TWEDistanceSpace();
    }

    @Parameterized.Parameters(name = "{0}")
    public static Collection<Object[]> data() throws Exception {
        return standardDatasets;
    }
}
