package tsml.classifiers.distance_based.distances.wdtw;

import com.beust.jcommander.internal.Lists;
import tsml.classifiers.distance_based.distances.DistanceMeasure;
import tsml.classifiers.distance_based.distances.transformed.BaseTransformDistanceMeasure;
import tsml.classifiers.distance_based.distances.transformed.TransformDistanceMeasure;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpace;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpaceBuilder;
import tsml.classifiers.distance_based.utils.collections.params.distribution.double_based.UniformDoubleDistribution;
import tsml.transformers.Derivative;
import weka.core.Instances;

import java.util.List;

import static tsml.classifiers.distance_based.utils.collections.CollectionUtils.newArrayList;
import static utilities.ArrayUtilities.unique;

public class WDTWDistanceConfigs {

    public static class WDTWSpaceBuilder implements ParamSpaceBuilder {

        @Override public ParamSpace build(final Instances data) {
            return buildWDTWSpace();
        }
    }

    public static class WDDTWSpaceBuilder implements ParamSpaceBuilder {

        @Override public ParamSpace build(final Instances data) {
            return buildWDDTWSpace();
        }
    }

    public static class ContinuousWDTWSpaceBuilder implements ParamSpaceBuilder {

        @Override public ParamSpace build(final Instances data) {
            return buildContinuousWDTWSpace();
        }
    }

    public static class ContinuousWDDTWSpaceBuilder implements ParamSpaceBuilder {

        @Override public ParamSpace build(final Instances data) {
            return buildContinuousWDDTWSpace();
        }
    }

    public static ParamSpace buildWDTWSpace() {
        return new ParamSpace()
                       .add(DistanceMeasure.DISTANCE_MEASURE_FLAG, newArrayList(new WDTWDistance()),
                               buildWDTWParams());
    }

    /**
     * This version includes full window. Original did not.
     * @return
     */
    public static ParamSpace buildWDTWParams() {
        double[] gValues = new double[101];
        for(int i = 0; i < gValues.length; i++) {
            gValues[i] = (double) i / 100;
        }
        List<Double> gValuesUnique = unique(gValues);
        ParamSpace params = new ParamSpace();
        params.add(WDTW.G_FLAG, gValuesUnique);
        return params;
    }

    /**
     * build WDDTW
     *
     * @return
     */
    public static TransformDistanceMeasure newWDDTWDistance() {
        return new BaseTransformDistanceMeasure("WDDTWDistance", Derivative.getGlobalCachedTransformer(), new WDTWDistance());
    }

    public static ParamSpace buildWDDTWSpace() {
        return new ParamSpace().add(DistanceMeasure.DISTANCE_MEASURE_FLAG, newArrayList(newWDDTWDistance()),
                buildWDTWParams());
    }

    public static ParamSpace buildContinuousWDTWParams() {
        final ParamSpace subSpace = new ParamSpace();
        subSpace.add(WDTW.G_FLAG, new UniformDoubleDistribution(0d, 1d));
        return subSpace;
    }

    public static ParamSpace buildContinuousWDTWSpace() {
        final ParamSpace space = new ParamSpace();
        space.add(DistanceMeasure.DISTANCE_MEASURE_FLAG, Lists.newArrayList(new WDTWDistance()),
                  buildContinuousWDTWParams());
        return space;
    }

    public static ParamSpace buildContinuousWDDTWSpace() {
        final ParamSpace space = new ParamSpace();
        space.add(DistanceMeasure.DISTANCE_MEASURE_FLAG, Lists.newArrayList(newWDDTWDistance()),
                  buildContinuousWDTWParams());
        return space;
    }
}
