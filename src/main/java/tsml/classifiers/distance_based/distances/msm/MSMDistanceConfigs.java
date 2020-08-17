package tsml.classifiers.distance_based.distances.msm;

import tsml.classifiers.distance_based.distances.DistanceMeasure;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpace;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpaceBuilder;
import tsml.classifiers.distance_based.utils.collections.params.distribution.double_based.DoubleDistribution;
import tsml.classifiers.distance_based.utils.collections.params.distribution.double_based.MultipleDoubleDistribution;
import weka.core.Instances;

import java.util.List;

import static tsml.classifiers.distance_based.utils.collections.CollectionUtils.newArrayList;
import static utilities.ArrayUtilities.unique;

public class MSMDistanceConfigs {
    public static class MSMSpaceBuilder implements ParamSpaceBuilder {

        @Override public ParamSpace build(final Instances data) {
            return buildMSMSpace();
        }
    }

    public static class ContinuousMSMSpaceBuilder implements ParamSpaceBuilder {
        @Override public ParamSpace build(final Instances data) {
            return buildContinuousMSMSpace();
        }
    }

    public static ParamSpace buildMSMSpace() {
        return new ParamSpace().add(DistanceMeasure.DISTANCE_MEASURE_FLAG, newArrayList(new MSMDistance()),
                buildMSMParams());
    }

    public static ParamSpace buildMSMParams() {
        double[] costValues = {
                // <editor-fold defaultstate="collapsed" desc="hidden for space">
                0.01,
                0.01375,
                0.0175,
                0.02125,
                0.025,
                0.02875,
                0.0325,
                0.03625,
                0.04,
                0.04375,
                0.0475,
                0.05125,
                0.055,
                0.05875,
                0.0625,
                0.06625,
                0.07,
                0.07375,
                0.0775,
                0.08125,
                0.085,
                0.08875,
                0.0925,
                0.09625,
                0.1,
                0.136,
                0.172,
                0.208,
                0.244,
                0.28,
                0.316,
                0.352,
                0.388,
                0.424,
                0.46,
                0.496,
                0.532,
                0.568,
                0.604,
                0.64,
                0.676,
                0.712,
                0.748,
                0.784,
                0.82,
                0.856,
                0.892,
                0.928,
                0.964,
                1,
                1.36,
                1.72,
                2.08,
                2.44,
                2.8,
                3.16,
                3.52,
                3.88,
                4.24,
                4.6,
                4.96,
                5.32,
                5.68,
                6.04,
                6.4,
                6.76,
                7.12,
                7.48,
                7.84,
                8.2,
                8.56,
                8.92,
                9.28,
                9.64,
                10,
                13.6,
                17.2,
                20.8,
                24.4,
                28,
                31.6,
                35.2,
                38.8,
                42.4,
                46,
                49.6,
                53.2,
                56.8,
                60.4,
                64,
                67.6,
                71.2,
                74.8,
                78.4,
                82,
                85.6,
                89.2,
                92.8,
                96.4,
                100// </editor-fold>
        };
        List<Double> costValuesUnique = unique(costValues);
        ParamSpace params = new ParamSpace();
        params.add(MSMDistance.C_FLAG, costValuesUnique);
        return params;
    }

    public static ParamSpace buildContinuousMSMParams() {
        DoubleDistribution costParams = new MultipleDoubleDistribution(newArrayList(0.01, 0.1, 1d, 10d, 100d));
        ParamSpace params = new ParamSpace();
        params.add(MSMDistance.C_FLAG, costParams);
        return params;
    }

    public static ParamSpace buildContinuousMSMSpace() {
        return new ParamSpace().add(DistanceMeasure.DISTANCE_MEASURE_FLAG, newArrayList(new MSMDistance()),
                buildContinuousMSMParams());
    }

}
