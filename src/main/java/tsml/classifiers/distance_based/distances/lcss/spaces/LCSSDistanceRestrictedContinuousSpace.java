package tsml.classifiers.distance_based.distances.lcss.spaces;

import tsml.classifiers.distance_based.distances.DistanceMeasure;
import tsml.classifiers.distance_based.distances.lcss.LCSSDistance;
import tsml.classifiers.distance_based.utils.collections.params.ParamMap;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpace;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpaceBuilder;
import tsml.data_containers.TimeSeriesInstances;

import static tsml.classifiers.distance_based.utils.collections.CollectionUtils.newArrayList;

public class LCSSDistanceRestrictedContinuousSpace implements ParamSpaceBuilder {
    @Override public ParamSpace build(final TimeSeriesInstances data) {
        return new ParamSpace(new ParamMap().add(DistanceMeasure.DISTANCE_MEASURE_FLAG, newArrayList(new LCSSDistance()), new LCSSDistanceRestrictedContinuousParams().build(data)));
    }
}
