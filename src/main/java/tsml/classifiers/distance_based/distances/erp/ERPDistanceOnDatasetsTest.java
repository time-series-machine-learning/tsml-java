package tsml.classifiers.distance_based.distances.erp;

import org.junit.runners.Parameterized;
import tsml.classifiers.distance_based.distances.DistanceMeasureOnDatasetsTest;
import tsml.classifiers.distance_based.distances.erp.spaces.ERPDistanceSpace;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpaceBuilder;

import java.util.Collection;

public class ERPDistanceOnDatasetsTest
        extends DistanceMeasureOnDatasetsTest {

    @Override public ParamSpaceBuilder getBuilder() {
        return new ERPDistanceSpace();
    }

    @Parameterized.Parameters(name = "{0}")
    public static Collection<Object[]> data() throws Exception {
        return standardDatasets;
    }
}
