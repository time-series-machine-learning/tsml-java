package tsml.classifiers.distance_based.distances.lcss;

import org.junit.runners.Parameterized;
import tsml.classifiers.distance_based.distances.DistanceMeasureOnDatasetsTest;
import tsml.classifiers.distance_based.distances.lcss.spaces.LCSSDistanceSpace;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpaceBuilder;

import java.util.Collection;

public class LCSSDistanceOnDatasetsTest
        extends DistanceMeasureOnDatasetsTest {

    @Override public ParamSpaceBuilder getBuilder() {
        return new LCSSDistanceSpace();
    }

    @Parameterized.Parameters(name = "{0}")
    public static Collection<Object[]> data() throws Exception {
        return standardDatasets;
    }
}
