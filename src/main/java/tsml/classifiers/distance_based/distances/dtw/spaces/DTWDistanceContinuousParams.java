package tsml.classifiers.distance_based.distances.dtw.spaces;

import tsml.classifiers.distance_based.utils.collections.params.ParamMap;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpace;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpaceBuilder;
import tsml.classifiers.distance_based.utils.collections.params.distribution.double_based.UniformDoubleDistribution;
import tsml.data_containers.TimeSeriesInstances;

import static tsml.classifiers.distance_based.distances.dtw.DTW.WINDOW_FLAG;

public class DTWDistanceContinuousParams implements ParamSpaceBuilder {
    @Override public ParamSpace build(final TimeSeriesInstances data) {
        return new ParamSpace(new ParamMap().add(WINDOW_FLAG, new UniformDoubleDistribution(0d, 1d)));
    }
}
