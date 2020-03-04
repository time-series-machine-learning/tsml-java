package tsml.classifiers.distance_based.distances.ddtw;

import org.junit.Test;
import tsml.classifiers.distance_based.distances.DistanceMeasureConfigs;
import tsml.classifiers.distance_based.distances.ReproducibleDistanceFunctionTest;
import tsml.classifiers.distance_based.utils.params.ParamSpace;
import weka.core.Debug.Random;
import weka.core.DistanceFunction;
import weka.core.Instances;

/**
 * Purpose: test DDTWDistance
 * <p>
 * Contributors: goastler
 */
public class DDTWDistanceTest extends ReproducibleDistanceFunctionTest {

    @Override
    protected DistanceFunction getDistanceFunction() {
        return new DDTWDistance();
    }

    @Override
    protected ParamSpace getDistanceFunctionParamSpace() {
        return DistanceMeasureConfigs.buildDdtwSpaceV1(data);
    }
}
