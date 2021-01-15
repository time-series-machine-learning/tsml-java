package tsml.classifiers.distance_based.distances.wdtw;

import org.junit.runners.Parameterized;
import tsml.classifiers.distance_based.distances.DistanceMeasureOnDatasetsTest;
import tsml.classifiers.distance_based.distances.wdtw.spaces.WDTWDistanceSpace;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpaceBuilder;

import java.util.Collection;

public class WDTWOnDatasetsTest
        extends DistanceMeasureOnDatasetsTest {

    @Override public ParamSpaceBuilder getBuilder() {
        return new WDTWDistanceSpace();
    }

    @Parameterized.Parameters(name = "{0}")
    public static Collection<Object[]> data() throws Exception {
        return standardDatasets;
    }
}
