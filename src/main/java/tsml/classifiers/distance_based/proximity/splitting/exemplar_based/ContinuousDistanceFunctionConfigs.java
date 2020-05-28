package tsml.classifiers.distance_based.proximity.splitting.exemplar_based;

import com.beust.jcommander.internal.Lists;
import experiments.data.DatasetLoading;
import java.util.Random;
import tsml.classifiers.distance_based.distances.DistanceMeasureable;
import tsml.classifiers.distance_based.distances.ddtw.DDTWDistance;
import tsml.classifiers.distance_based.distances.dtw.DTW;
import tsml.classifiers.distance_based.distances.dtw.DTWDistance;
import tsml.classifiers.distance_based.distances.erp.ERPDistance;
import tsml.classifiers.distance_based.distances.lcss.LCSSDistance;
import tsml.classifiers.distance_based.distances.msm.MSMDistance;
import tsml.classifiers.distance_based.distances.wddtw.WDDTWDistance;
import tsml.classifiers.distance_based.distances.wdtw.WDTW;
import tsml.classifiers.distance_based.distances.wdtw.WDTWDistance;
import tsml.classifiers.distance_based.proximity.RandomSource;
import tsml.classifiers.distance_based.utils.params.ParamSpace;
import tsml.classifiers.distance_based.utils.params.distribution.DoubleToIntDistributionAdapter;
import tsml.classifiers.distance_based.utils.params.distribution.UniformDistribution;
import utilities.InstanceTools;
import utilities.StatisticalUtilities;
import utilities.Utilities;
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
        subSpace.add(DTW.getWarpingWindowFlag(), new DoubleToIntDistributionAdapter(new UniformDistribution(0,
            (int) (((double) data.numAttributes() + 1) / 4))));
        return subSpace;
    }

    public static ParamSpace buildDtwSpace(Instances data) {
        final ParamSpace space = new ParamSpace();
        space.add(DistanceMeasureable.getDistanceFunctionFlag(), Lists.newArrayList(new DTWDistance()),
            buildDtwParams(data));
        return space;
    }

    public static ParamSpace buildDdtwSpace(Instances data) {
        final ParamSpace space = new ParamSpace();
        space.add(DistanceMeasureable.getDistanceFunctionFlag(), Lists.newArrayList(new DDTWDistance()),
            buildDtwParams(data));
        return space;
    }

    public static ParamSpace buildErpParams(Instances data) {
        final double std = StatisticalUtilities.pStdDev(data);
        final ParamSpace subSpace = new ParamSpace();
        // pf implements this as randInt(len / 4 + 1), so range is from 0 to len / 4 inclusively
        // above doesn't consider class value, so -1 from len
        subSpace.add(ERPDistance.getBandSizeFlag(), new DoubleToIntDistributionAdapter(new UniformDistribution(0,
            (int) (((double) data.numAttributes() - 1) / 4))));
        subSpace.add(ERPDistance.getPenaltyFlag(), new UniformDistribution(0.2 * std, std));
        return subSpace;
    }

    public static ParamSpace buildErpSpace(Instances data) {
        final ParamSpace space = new ParamSpace();
        space.add(DistanceMeasureable.getDistanceFunctionFlag(), Lists.newArrayList(new ERPDistance()),
            buildErpParams(data));
        return space;
    }

    public static ParamSpace buildLcssParams(Instances data) {
        final double std = StatisticalUtilities.pStdDev(data);
        final ParamSpace subSpace = new ParamSpace();
        // pf implements this as randInt((len + 1) / 4), so range is from 0 to (len + 1) / 4 - 1 inclusively.
        // above doesn't consider class value, so -1 from len
        subSpace.add(LCSSDistance.getDeltaFlag(), new DoubleToIntDistributionAdapter(new UniformDistribution(0,
            (int) ((double) data.numAttributes() / 4) - 1)));
        subSpace.add(LCSSDistance.getEpsilonFlag(), new UniformDistribution(0.2 * std, std));
        return subSpace;
    }

    public static ParamSpace buildLcssSpace(Instances data) {
        final ParamSpace space = new ParamSpace();
        space.add(DistanceMeasureable.getDistanceFunctionFlag(), Lists.newArrayList(new LCSSDistance()),
            buildLcssParams(data));
        return space;
    }

    public static ParamSpace buildWdtwParams() {
        final ParamSpace subSpace = new ParamSpace();
        subSpace.add(WDTW.getGFlag(), new UniformDistribution(0, 1));
        return subSpace;
    }

    public static ParamSpace buildWdtwSpace() {
        final ParamSpace space = new ParamSpace();
        space.add(DistanceMeasureable.getDistanceFunctionFlag(), Lists.newArrayList(new WDTWDistance()),
            buildWdtwParams());
        return space;
    }


    public static ParamSpace buildWddtwSpace() {
        final ParamSpace space = new ParamSpace();
        space.add(DistanceMeasureable.getDistanceFunctionFlag(), Lists.newArrayList(new WDDTWDistance()),
            buildWdtwParams());
        return space;
    }

    // todo unit test these just to make sure param flags match up and actually set the param when applied
}
