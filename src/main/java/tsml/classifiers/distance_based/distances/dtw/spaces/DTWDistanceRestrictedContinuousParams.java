package tsml.classifiers.distance_based.distances.dtw.spaces;

import tsml.classifiers.distance_based.utils.collections.params.ParamMap;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpace;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpaceBuilder;
import tsml.classifiers.distance_based.utils.collections.params.distribution.double_based.UniformDoubleDistribution;
import tsml.data_containers.TimeSeriesInstances;

import static tsml.classifiers.distance_based.distances.dtw.DTW.WINDOW_FLAG;

public class DTWDistanceRestrictedContinuousParams implements ParamSpaceBuilder {
    @Override public ParamSpace build(final TimeSeriesInstances data) {
        final ParamMap subSpace = new ParamMap();
        subSpace.add(WINDOW_FLAG, new UniformDoubleDistribution(0d, 0.25d));
        return new ParamSpace(subSpace);
    }
}
