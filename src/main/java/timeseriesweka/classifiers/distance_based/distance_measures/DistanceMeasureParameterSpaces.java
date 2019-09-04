package timeseriesweka.classifiers.distance_based.distance_measures;

import evaluation.tuning.ParameterSpace;
import utilities.ArrayUtilities;
import utilities.StatisticalUtilities;
import weka.core.Instances;

import static timeseriesweka.classifiers.distance_based.distance_measures.DistanceMeasure.DISTANCE_MEASURE_KEY;
import static timeseriesweka.classifiers.distance_based.distance_measures.Dtw.WARPING_WINDOW_KEY;

public class DistanceMeasureParameterSpaces {
    private DistanceMeasureParameterSpaces() {

    }

    public static ParameterSpace buildEdParameterSpace() {
        ParameterSpace parameterSpace = new ParameterSpace();
        parameterSpace.addParameter(DISTANCE_MEASURE_KEY, Dtw.NAME);
        parameterSpace.addParameter(WARPING_WINDOW_KEY, 0);
        return parameterSpace;
    }

    public static ParameterSpace buildFullDtwParameterSpace() {
        ParameterSpace parameterSpace = new ParameterSpace();
        parameterSpace.addParameter(DISTANCE_MEASURE_KEY, Dtw.NAME);
        parameterSpace.addParameter(Dtw.WARPING_WINDOW_KEY, -1);
        return parameterSpace;
    }

    public static ParameterSpace buildDtwParameterSpace(Instances instances) {
        ParameterSpace parameterSpace = new ParameterSpace();
        parameterSpace.addParameter(DISTANCE_MEASURE_KEY, Dtw.NAME);
        parameterSpace.addParameter(Dtw.WARPING_WINDOW_KEY, ArrayUtilities.incrementalRange(0, instances.numAttributes() - 1, 100));
        return parameterSpace;
    }

    public static ParameterSpace buildAllDtwParameterSpace(Instances instances) {
        ParameterSpace parameterSpace = buildDtwParameterSpace(instances);
        parameterSpace.addAll(buildFullDtwParameterSpace());
        return parameterSpace;
    }

    public static ParameterSpace buildDdtwParameterSpace(Instances instances) {
        ParameterSpace parameterSpace = buildDtwParameterSpace(instances);
        parameterSpace.addParameter(DISTANCE_MEASURE_KEY, Ddtw.NAME);
        return parameterSpace;
    }

    public static ParameterSpace buildFullDdtwParameterSpace() {
        ParameterSpace parameterSpace = buildFullDtwParameterSpace();
        parameterSpace.addParameter(DISTANCE_MEASURE_KEY, Ddtw.NAME);
        return parameterSpace;
    }

    public static ParameterSpace buildWdtwParameterSpace() {
        double[] gValues = new double[100];
        for(int i = 0; i < gValues.length; i++) {
            gValues[i] = (double) i / gValues.length;
        }
        ParameterSpace parameterSpace = new ParameterSpace();
        parameterSpace.addParameter(Wdtw.DISTANCE_MEASURE_KEY, Wdtw.NAME);
        parameterSpace.addParameter(Wdtw.WEIGHT_KEY, gValues);
        return parameterSpace;
    }

    public static ParameterSpace buildWddtwParameterSpace() {
        ParameterSpace parameterSpace = buildWdtwParameterSpace();
        parameterSpace.putParameter(DISTANCE_MEASURE_KEY, Wddtw.NAME);
        return parameterSpace;
    }

    public static ParameterSpace buildTwedParameterSpace() {

        double[] nuValues = {
                // <editor-fold defaultstate="collapsed" desc="hidden for space">
                0.00001,
                0.0001,
                0.0005,
                0.001,
                0.005,
                0.01,
                0.05,
                0.1,
                0.5,
                1,// </editor-fold>
        };
        double[] lambdaValues = {
                // <editor-fold defaultstate="collapsed" desc="hidden for space">
                0,
                0.011111111,
                0.022222222,
                0.033333333,
                0.044444444,
                0.055555556,
                0.066666667,
                0.077777778,
                0.088888889,
                0.1,// </editor-fold>
        };
        ParameterSpace parameterSpace = new ParameterSpace();
        parameterSpace.addParameter(DISTANCE_MEASURE_KEY, Twed.NAME);
        parameterSpace.addParameter(Twed.NU_KEY, nuValues);
        parameterSpace.addParameter(Twed.LAMBDA_KEY, lambdaValues);
        return parameterSpace;
    }

    public static ParameterSpace buildMsmParameterSpace() {
        double[] penaltyValues = {
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
        ParameterSpace parameterSpace = new ParameterSpace();
        parameterSpace.addParameter(DISTANCE_MEASURE_KEY, Msm.NAME);
        parameterSpace.addParameter(Msm.COST_KEY, penaltyValues);
        return parameterSpace;
    }

    public static ParameterSpace buildLcssParameterSpace(Instances instances) {
        double std = StatisticalUtilities.pStdDev(instances);
        double stdFloor = std*0.2;
        double[] epsilonValues = ArrayUtilities.incrementalRange(stdFloor, std, 10);
        int[] deltaValues = ArrayUtilities.incrementalRange(0, (instances.numAttributes() - 1) / 4, 10);
        ParameterSpace parameterSpace = new ParameterSpace();
        parameterSpace.addParameter(DISTANCE_MEASURE_KEY, Lcss.NAME);
        parameterSpace.addParameter(Lcss.DELTA_KEY, deltaValues);
        parameterSpace.addParameter(Lcss.EPSILON_KEY, epsilonValues);
        return parameterSpace;
    }

    public static ParameterSpace buildErpParameterSpace(Instances instances) {
        double std = StatisticalUtilities.pStdDev(instances);
        double stdFloor = std*0.2;
        int[] bandSizeValues = ArrayUtilities.incrementalRange(0, (instances.numAttributes() - 1) / 4, 10);
        double[] penaltyValues = ArrayUtilities.incrementalRange(stdFloor, std, 10);
        ParameterSpace parameterSpace = new ParameterSpace();
        parameterSpace.addParameter(DISTANCE_MEASURE_KEY, Erp.NAME);
        parameterSpace.addParameter(Erp.BAND_SIZE_KEY, bandSizeValues);
        parameterSpace.addParameter(Erp.PENALTY_KEY, penaltyValues);
        return parameterSpace;
    }
}
