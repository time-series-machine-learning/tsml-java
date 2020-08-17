package tsml.classifiers.distance_based.distances.lcss;

import tsml.classifiers.distance_based.distances.DistanceMeasure;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpace;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpaceBuilder;
import tsml.classifiers.distance_based.utils.collections.params.distribution.double_based.UniformDoubleDistribution;
import tsml.classifiers.distance_based.utils.collections.params.distribution.int_based.UniformIntDistribution;
import utilities.StatisticalUtilities;
import weka.core.Instances;

import java.util.List;

import static tsml.classifiers.distance_based.utils.collections.CollectionUtils.newArrayList;
import static utilities.ArrayUtilities.range;
import static utilities.ArrayUtilities.unique;

public class LCSSDistanceConfigs {

    public static class LCSSSpaceBuilder implements ParamSpaceBuilder {

        @Override public ParamSpace build(final Instances data) {
            return buildLCSSSpace(data);
        }
    }

    public static class ContinuousLCSSSpaceBuilder implements ParamSpaceBuilder {

        @Override public ParamSpace build(final Instances data) {
            return buildContinuousLCSSSpace(data);
        }
    }

    public static class RestrictedContinuousLCSSSpaceBuilder implements ParamSpaceBuilder {

        @Override public ParamSpace build(final Instances data) {
            return buildRestrictedContinuousLCSSSpace(data);
        }
    }

    public static ParamSpace buildLCSSSpace(Instances instances) {
        return new ParamSpace().add(DistanceMeasure.DISTANCE_MEASURE_FLAG, newArrayList(new LCSSDistance()), buildLCSSParams(instances));
    }

    public static ParamSpace buildLCSSParams(Instances instances) {
        double std = StatisticalUtilities.pStdDev(instances);
        double stdFloor = std * 0.2;
        double[] epsilonValues = range(stdFloor, std, 10);
        int[] deltaValues = range(0, (instances.numAttributes() - 1) / 4, 10);
        List<Double> epsilonValuesUnique = unique(epsilonValues);
        List<Integer> deltaValuesUnique = unique(deltaValues);
        ParamSpace params = new ParamSpace();
        params.add(LCSSDistance.EPSILON_FLAG, epsilonValuesUnique);
        params.add(LCSSDistance.WINDOW_SIZE_FLAG, deltaValuesUnique);
        return params;
    }

    public static ParamSpace buildRestrictedContinuousLCSSParams(Instances data) {
        final double std = StatisticalUtilities.pStdDev(data);
        final ParamSpace subSpace = new ParamSpace();
        subSpace.add(LCSSDistance.EPSILON_FLAG, new UniformDoubleDistribution(0.2 * std, std));
        // pf implements this as randInt((len + 1) / 4), so range is from 0 to (len + 1) / 4 - 1 inclusively.
        // above doesn't consider class value, so -1 from len
        subSpace.add(LCSSDistance.WINDOW_SIZE_FLAG, new UniformIntDistribution(0,
            data.numAttributes() / 4 - 1));
        return subSpace;
    }

    public static ParamSpace buildRestrictedContinuousLCSSSpace(Instances data) {
        final ParamSpace space = new ParamSpace();
        space.add(DistanceMeasure.DISTANCE_MEASURE_FLAG, newArrayList(new LCSSDistance()),
                  buildRestrictedContinuousLCSSParams(data));
        return space;
    }

    public static ParamSpace buildContinuousLCSSParams(Instances data) {
        final double std = StatisticalUtilities.pStdDev(data);
        final ParamSpace subSpace = new ParamSpace();
        subSpace.add(LCSSDistance.EPSILON_FLAG, new UniformDoubleDistribution(0.02 * std, std));
        subSpace.add(LCSSDistance.WINDOW_SIZE_FLAG, new UniformIntDistribution(0, data.numAttributes() - 1 - 1));
        return subSpace;
    }

    public static ParamSpace buildContinuousLCSSSpace(Instances data) {
        final ParamSpace space = new ParamSpace();
        space.add(DistanceMeasure.DISTANCE_MEASURE_FLAG, newArrayList(new LCSSDistance()), buildContinuousLCSSParams(data));
        return space;
    }
}
