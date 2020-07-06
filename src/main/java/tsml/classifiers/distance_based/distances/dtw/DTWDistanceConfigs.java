package tsml.classifiers.distance_based.distances.dtw;

import com.beust.jcommander.internal.Lists;
import tsml.classifiers.distance_based.distances.DistanceMeasure;
import tsml.classifiers.distance_based.distances.WarpingDistanceMeasure;
import tsml.classifiers.distance_based.distances.transformed.BaseTransformDistanceMeasure;
import tsml.classifiers.distance_based.distances.transformed.TransformDistanceMeasure;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpace;
import tsml.classifiers.distance_based.utils.collections.params.distribution.int_based.UniformIntDistribution;
import tsml.transformers.Derivative;
import weka.core.Instances;

import static tsml.classifiers.distance_based.utils.collections.CollectionUtils.newArrayList;
import static utilities.ArrayUtilities.range;
import static utilities.ArrayUtilities.unique;

public class DTWDistanceConfigs {
    /**
     * param space containing full derivative DTW params (i.e. full window)
     *
     * @return
     */
    public static ParamSpace buildDdtwFullWindowSpace() {
        return new ParamSpace().add(DistanceMeasure.DISTANCE_MEASURE_FLAG,
                newArrayList(newDDTWDistance()),
                buildDtwFullWindowParams());
    }

    /**
     * build DDTW
     *
     * @return
     */
    public static TransformDistanceMeasure newDDTWDistance() {
        return new BaseTransformDistanceMeasure("DDTWDistance", Derivative.getGlobalCachedTransformer(), new DTWDistance());
    }

    /**
     * Build DTW space with corresponding params. This includes ED and Full DTW
     * @param instances
     * @return
     */
    public static ParamSpace buildDtwSpace(Instances instances) {
        return new ParamSpace().add(DistanceMeasure.DISTANCE_MEASURE_FLAG, newArrayList(new DTWDistance()),
                buildDtwParams(instances));
    }

    /**
     * Build DTW params. This includes ED and Full DTW
     * @param instances
     * @return
     */
    public static ParamSpace buildDtwParams(Instances instances) {
        return new ParamSpace()
                       .add(WarpingDistanceMeasure.WINDOW_SIZE_FLAG, unique(range(0,
                               instances.numAttributes() - 1, 100)));
    }

    /**
     * Same as DTW version but with derivative. This includes ED and Full DTW (deriv versions)
     * @param instances
     * @return
     */
    public static ParamSpace buildDdtwSpace(Instances instances) {
        return new ParamSpace().add(DistanceMeasure.DISTANCE_MEASURE_FLAG, newArrayList(newDDTWDistance()),
                buildDtwParams(instances));
    }

    public static ParamSpace buildDtwParamsRestrictedContinuous(Instances data) {
        final ParamSpace subSpace = new ParamSpace();
        // pf implements this as randInt((len + 1) / 4), so range is from 0 to (len + 1) / 4 - 1 inclusively.
        // above doesn't consider class value, so -1 from len
        subSpace.add(WarpingDistanceMeasure.WINDOW_SIZE_FLAG, new UniformIntDistribution(0,
            (data.numAttributes()) / 4 - 1));
        return subSpace;
    }

    public static ParamSpace buildDtwSpaceRestrictedContinuous(Instances data) {
        final ParamSpace space = new ParamSpace();
        space.add(DistanceMeasure.DISTANCE_MEASURE_FLAG, Lists.newArrayList(new DTWDistance()),
                  buildDtwParamsRestrictedContinuous(data));
        return space;
    }

    public static ParamSpace buildDdtwSpaceRestrictedContinuous(Instances data) {
        final ParamSpace space = new ParamSpace();
        space.add(DistanceMeasure.DISTANCE_MEASURE_FLAG, Lists.newArrayList(newDDTWDistance()),
                  buildDtwParamsRestrictedContinuous(data));
        return space;
    }

    /**
     * param space containing full DTW
     *
     * @return
     */
    public static ParamSpace buildDtwFullWindowSpace() {
        return new ParamSpace().add(DistanceMeasure.DISTANCE_MEASURE_FLAG, newArrayList(new DTWDistance()),
                buildDtwFullWindowParams());
    }

    /**
     * param space containing full DTW params (i.e. full window)
     *
     * @return
     */
    public static ParamSpace buildDtwFullWindowParams() {
        ParamSpace params = new ParamSpace();
        params.add(WarpingDistanceMeasure.WINDOW_SIZE_FLAG, newArrayList(-1));
        return params;
    }
}
