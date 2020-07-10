package tsml.classifiers.distance_based.distances.ed;

import tsml.classifiers.distance_based.distances.DistanceMeasure;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpace;

import static tsml.classifiers.distance_based.utils.collections.CollectionUtils.newArrayList;

public class EDistanceConfigs {
    /**
     * param space containing ED
     *
     * @return
     */
    public static ParamSpace buildEdSpace() {
        return new ParamSpace().add(DistanceMeasure.DISTANCE_MEASURE_FLAG, newArrayList(new EDistance()));
    }
}
