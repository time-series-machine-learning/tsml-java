package tsml.classifiers.distance_based.distances.dtw.spaces;

import tsml.classifiers.distance_based.distances.dtw.DTWDistance;
import tsml.classifiers.distance_based.distances.transformed.BaseTransformDistanceMeasure;
import tsml.classifiers.distance_based.distances.transformed.TransformDistanceMeasure;
import tsml.classifiers.distance_based.utils.collections.params.ParamMap;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpace;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpaceBuilder;
import tsml.data_containers.TimeSeriesInstances;
import tsml.transformers.Derivative;

import static tsml.classifiers.distance_based.distances.DistanceMeasure.DISTANCE_MEASURE_FLAG;
import static tsml.classifiers.distance_based.utils.collections.CollectionUtils.newArrayList;

public class DDTWDistanceSpace implements ParamSpaceBuilder {
    @Override public ParamSpace build(final TimeSeriesInstances data) {
        return new ParamSpace(new ParamMap().add(DISTANCE_MEASURE_FLAG, newArrayList(newDDTWDistance()), new DTWDistanceSpace().build(data)));
    }
    
    /**
     * build DDTW
     *
     * @return
     */
    public static TransformDistanceMeasure newDDTWDistance() {
        return new BaseTransformDistanceMeasure("DDTWDistance", new Derivative(), new DTWDistance());
    }
}
