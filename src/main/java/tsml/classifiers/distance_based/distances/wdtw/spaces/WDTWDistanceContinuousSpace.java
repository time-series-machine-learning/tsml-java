package tsml.classifiers.distance_based.distances.wdtw.spaces;

import tsml.classifiers.distance_based.distances.DistanceMeasure;
import tsml.classifiers.distance_based.distances.wdtw.WDTWDistance;
import tsml.classifiers.distance_based.utils.collections.params.ParamMap;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpace;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpaceBuilder;
import tsml.data_containers.TimeSeriesInstances;

import static tsml.classifiers.distance_based.utils.collections.CollectionUtils.newArrayList;

public class WDTWDistanceContinuousSpace implements ParamSpaceBuilder {

    @Override public ParamSpace build(final TimeSeriesInstances data) {
        return new ParamSpace(new ParamMap().add(DistanceMeasure.DISTANCE_MEASURE_FLAG, newArrayList(new WDTWDistance()),
                new WDTWDistanceContinuousParams().build(data)));
    }
}
