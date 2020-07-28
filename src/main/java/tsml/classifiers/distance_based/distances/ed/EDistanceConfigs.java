package tsml.classifiers.distance_based.distances.ed;

import tsml.classifiers.distance_based.distances.DistanceMeasure;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpace;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpaceBuilder;
import weka.core.Instances;

import static tsml.classifiers.distance_based.utils.collections.CollectionUtils.newArrayList;

public class EDistanceConfigs {
    /**
     * param space containing ED
     *
     * @return
     */
    public static ParamSpace buildEDSpace() {
        return new ParamSpace().add(DistanceMeasure.DISTANCE_MEASURE_FLAG, newArrayList(new EDistance()));
    }

    public static class EDSpaceBuilder implements ParamSpaceBuilder {

        @Override public ParamSpace build(final Instances data) {
            return buildEDSpace();
        }
    }
}
