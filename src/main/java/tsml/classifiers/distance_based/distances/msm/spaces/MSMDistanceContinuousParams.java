package tsml.classifiers.distance_based.distances.msm.spaces;

import tsml.classifiers.distance_based.distances.msm.MSMDistance;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpace;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpaceBuilder;
import tsml.classifiers.distance_based.utils.collections.params.distribution.CompositeDistribution;
import tsml.classifiers.distance_based.utils.collections.params.distribution.Distribution;
import tsml.data_containers.TimeSeriesInstances;

import java.util.List;

import static tsml.classifiers.distance_based.utils.collections.CollectionUtils.newArrayList;
import static utilities.ArrayUtilities.unique;

public class MSMDistanceContinuousParams implements ParamSpaceBuilder {
    @Override public ParamSpace build(final TimeSeriesInstances data) {
        Distribution<Double> costParams = CompositeDistribution
                                                  .newUniformDoubleCompositeFromRange(newArrayList(0.01, 0.1, 1d, 10d, 100d));
        ParamSpace params = new ParamSpace();
        params.add(MSMDistance.C_FLAG, costParams);
        return params;
    }
}
