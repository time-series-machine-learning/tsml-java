package tsml.classifiers.distance_based.distances.wdtw.spaces;

import tsml.classifiers.distance_based.distances.wdtw.WDTW;
import tsml.classifiers.distance_based.utils.collections.params.ParamMap;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpace;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpaceBuilder;
import tsml.classifiers.distance_based.utils.collections.params.distribution.double_based.UniformDoubleDistribution;
import tsml.data_containers.TimeSeriesInstances;

import static utilities.ArrayUtilities.unique;

public class WDTWDistanceContinuousParams implements ParamSpaceBuilder {

    @Override public ParamSpace build(final TimeSeriesInstances data) {
        final ParamMap subSpace = new ParamMap();
        subSpace.add(WDTW.G_FLAG, new UniformDoubleDistribution(0d, 1d));
        return new ParamSpace(subSpace);
    }
}
