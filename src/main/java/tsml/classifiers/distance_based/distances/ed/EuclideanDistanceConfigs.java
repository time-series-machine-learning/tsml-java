package tsml.classifiers.distance_based.distances.ed;

import tsml.classifiers.distance_based.distances.DistanceMeasure;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpace;
import weka.core.EuclideanDistance;

import static tsml.classifiers.distance_based.utils.collections.CollectionUtils.newArrayList;

public class EuclideanDistanceConfigs {
    /**
     * param space containing ED
     *
     * @return
     */
    public static ParamSpace buildEdSpace() {
        return new ParamSpace().add(DistanceMeasure.DISTANCE_MEASURE_FLAG, newArrayList(new EuclideanDistance()));
    }
}
