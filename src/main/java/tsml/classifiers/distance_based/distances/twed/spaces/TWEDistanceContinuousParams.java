package tsml.classifiers.distance_based.distances.twed.spaces;

import tsml.classifiers.distance_based.distances.twed.TWEDistance;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpace;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpaceBuilder;
import tsml.classifiers.distance_based.utils.collections.params.distribution.CompositeDistribution;
import tsml.classifiers.distance_based.utils.collections.params.distribution.Distribution;
import tsml.classifiers.distance_based.utils.collections.params.distribution.double_based.UniformDoubleDistribution;
import tsml.data_containers.TimeSeriesInstances;

import java.util.List;

import static tsml.classifiers.distance_based.utils.collections.CollectionUtils.newArrayList;
import static utilities.ArrayUtilities.unique;

public class TWEDistanceContinuousParams implements ParamSpaceBuilder {
    @Override public ParamSpace build(final TimeSeriesInstances data) {
        Distribution<Double> nuDistribution = CompositeDistribution.newUniformDoubleCompositeFromRange(newArrayList(0.00001,
                0.0001,
                0.0005,
                0.001,
                0.005,
                0.01,
                0.05,
                0.1,
                0.5,
                1d));
        UniformDoubleDistribution lambdaDistribution = new UniformDoubleDistribution();
        ParamSpace params = new ParamSpace();
        params.add(TWEDistance.LAMBDA_FLAG, lambdaDistribution);
        params.add(TWEDistance.NU_FLAG, nuDistribution);
        return params;
    }
}
