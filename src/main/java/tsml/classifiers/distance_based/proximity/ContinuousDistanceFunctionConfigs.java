package tsml.classifiers.distance_based.proximity;

import com.beust.jcommander.internal.Lists;
import experiments.data.DatasetLoading;
import tsml.classifiers.distance_based.distances.DistanceMeasureable;
import tsml.classifiers.distance_based.distances.WarpingDistanceMeasure;
import tsml.classifiers.distance_based.distances.ddtw.DDTWDistance;
import tsml.classifiers.distance_based.distances.dtw.DTWDistance;
import tsml.classifiers.distance_based.distances.erp.ERPDistance;
import tsml.classifiers.distance_based.distances.lcss.LCSSDistance;
import tsml.classifiers.distance_based.distances.wddtw.WDDTWDistance;
import tsml.classifiers.distance_based.distances.wdtw.WDTW;
import tsml.classifiers.distance_based.distances.wdtw.WDTWDistance;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpace;
import tsml.classifiers.distance_based.utils.collections.params.distribution.double_based.DoubleDistribution;
import tsml.classifiers.distance_based.utils.collections.params.distribution.double_based.UniformDoubleDistribution;
import tsml.classifiers.distance_based.utils.collections.params.distribution.int_based.UniformIntDistribution;
import utilities.StatisticalUtilities;
import weka.core.Instances;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class ContinuousDistanceFunctionConfigs {

    private ContinuousDistanceFunctionConfigs() {

    }

    public static void main(String[] args) throws Exception {
        buildErpParams(DatasetLoading.sampleGunPoint(0)[0]);
    }

    public static ParamSpace buildDtwParams(Instances data) {
        final ParamSpace subSpace = new ParamSpace();
        // pf implements this as randInt((len + 1) / 4), so range is from 0 to (len + 1) / 4 - 1 inclusively.
        // above doesn't consider class value, so -1 from len
        subSpace.add(WarpingDistanceMeasure.WINDOW_SIZE_FLAG, new UniformIntDistribution(0,
            (data.numAttributes()) / 4 - 1));
        return subSpace;
    }

    public static ParamSpace buildDtwSpace(Instances data) {
        final ParamSpace space = new ParamSpace();
        space.add(DistanceMeasureable.DISTANCE_MEASURE_FLAG, Lists.newArrayList(new DTWDistance()),
            buildDtwParams(data));
        return space;
    }

    public static ParamSpace buildDdtwSpace(Instances data) {
        final ParamSpace space = new ParamSpace();
        space.add(DistanceMeasureable.DISTANCE_MEASURE_FLAG, Lists.newArrayList(new DDTWDistance()),
            buildDtwParams(data));
        return space;
    }

    public static ParamSpace buildErpParams(Instances data) {
        final double std = StatisticalUtilities.pStdDev(data);
        final ParamSpace subSpace = new ParamSpace();
        subSpace.add(ERPDistance.G_FLAG, new DoubleDistribution(0,1) {

            @Override
            public Double sample() {
                return getRandom().nextDouble() * 0.8 * std + 0.2 * std;
            }
        });
//        subSpace.add(ERPDistance.getPenaltyFlag(), new UniformDoubleDistribution(0.2 * std, std));
        // pf implements this as randInt(len / 4 + 1), so range is from 0 to len / 4 inclusively
        // above doesn't consider class value, so -1 from len
        subSpace.add(ERPDistance.WINDOW_SIZE_FLAG, new UniformIntDistribution(0,
            (data.numAttributes() - 1) / 4));
        return subSpace;
    }

    public static ParamSpace buildErpSpace(Instances data) {
        final ParamSpace space = new ParamSpace();
        space.add(DistanceMeasureable.DISTANCE_MEASURE_FLAG, Lists.newArrayList(new ERPDistance()),
            buildErpParams(data));
        return space;
    }

    public static ParamSpace buildLcssParams(Instances data) {
        final double std = StatisticalUtilities.pStdDev(data);
        final ParamSpace subSpace = new ParamSpace();
        subSpace.add(LCSSDistance.EPSILON_FLAG, new UniformDoubleDistribution(0.2 * std, std));
        // pf implements this as randInt((len + 1) / 4), so range is from 0 to (len + 1) / 4 - 1 inclusively.
        // above doesn't consider class value, so -1 from len
        subSpace.add(LCSSDistance.WINDOW_SIZE_FLAG, new UniformIntDistribution(0,
            data.numAttributes() / 4 - 1));
        return subSpace;
    }

    public static ParamSpace buildLcssSpace(Instances data) {
        final ParamSpace space = new ParamSpace();
        space.add(DistanceMeasureable.DISTANCE_MEASURE_FLAG, Lists.newArrayList(new LCSSDistance()),
            buildLcssParams(data));
        return space;
    }

    public static ParamSpace buildWdtwParams() {
        final ParamSpace subSpace = new ParamSpace();
        subSpace.add(WDTW.G_FLAG, new UniformDoubleDistribution(0d, 1d));
        return subSpace;
    }

    public static ParamSpace buildWdtwSpace() {
        final ParamSpace space = new ParamSpace();
        space.add(DistanceMeasureable.DISTANCE_MEASURE_FLAG, Lists.newArrayList(new WDTWDistance()),
            buildWdtwParams());
        return space;
    }


    public static ParamSpace buildWddtwSpace() {
        final ParamSpace space = new ParamSpace();
        space.add(DistanceMeasureable.DISTANCE_MEASURE_FLAG, Lists.newArrayList(new WDDTWDistance()),
            buildWdtwParams());
        return space;
    }

}
