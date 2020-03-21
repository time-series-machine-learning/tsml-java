package tsml.classifiers.distance_based.proximity.splitting.exemplar_based;

import com.beust.jcommander.internal.Lists;
import java.util.Random;
import tsml.classifiers.distance_based.distances.DistanceMeasureable;
import tsml.classifiers.distance_based.distances.ddtw.DDTWDistance;
import tsml.classifiers.distance_based.distances.dtw.DTW;
import tsml.classifiers.distance_based.distances.dtw.DTWDistance;
import tsml.classifiers.distance_based.distances.erp.ERPDistance;
import tsml.classifiers.distance_based.distances.lcss.LCSSDistance;
import tsml.classifiers.distance_based.distances.msm.MSMDistance;
import tsml.classifiers.distance_based.distances.wddtw.WDDTWDistance;
import tsml.classifiers.distance_based.distances.wdtw.WDTWDistance;
import tsml.classifiers.distance_based.proximity.RandomSource;
import tsml.classifiers.distance_based.utils.params.ParamSpace;
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

    public static ParamSpace buildDtwSpace(Instances data) {
        final ParamSpace space = new ParamSpace();
        final ParamSpace subSpace = new ParamSpace();
        subSpace.add(DTW.getWarpingWindowFlag(), new UniformDistribution(0,
            (int) (((double) data.numAttributes() + 1) / 4)));
        space.add(DistanceMeasureable.getDistanceFunctionFlag(), new DTWDistance(), subSpace);
        return space;
    }

    public static ParamSpace buildDdtwSpace(Instances data) {
        final ParamSpace space = new ParamSpace();
        final ParamSpace subSpace = new ParamSpace();
        subSpace.add(DTW.getWarpingWindowFlag(), new UniformDistribution(0, (int) (((double) data.numAttributes() + 1) / 4)));
        space.add(DistanceMeasureable.getDistanceFunctionFlag(), new DDTWDistance(), subSpace);
        return space;
    }

    public static ParamSpace buildErpSpace(Instances data) {
        double std = StatisticalUtilities.pStdDev(data);
        final ParamSpace space = new ParamSpace();
        final ParamSpace subSpace = new ParamSpace();
        subSpace.add(ERPDistance.getBandSizeFlag(), new UniformDistribution(0, (int) (((double) data.numAttributes() + 1) / 4)));
        subSpace.add(ERPDistance.getPenaltyFlag(), new UniformDistribution(0.2 * std, std));
        space.add(DistanceMeasureable.getDistanceFunctionFlag(), new DDTWDistance(), subSpace);
        return space;
    }

    public static ParamSpace buildLcssSpace(Instances data) {
        double std = StatisticalUtilities.pStdDev(data);
        final ParamSpace space = new ParamSpace();
        final ParamSpace subSpace = new ParamSpace();
        subSpace.add(LCSSDistance.getDeltaFlag(), new UniformDistribution(0,
            (int) (((double) data.numAttributes() + 1) / 4)));
        subSpace.add(LCSSDistance.getEpsilonFlag(), new UniformDistribution(0.2 * std, std));
        space.add(DistanceMeasureable.getDistanceFunctionFlag(), new DDTWDistance(), subSpace);
        return space;
    }

    public static ParamSpace buildWdtwSpace(Instances data) {
        final ParamSpace space = new ParamSpace();
        final ParamSpace subSpace = new ParamSpace();
        subSpace.add(DTW.getWarpingWindowFlag(), new UniformDistribution(0, 1));
        space.add(DistanceMeasureable.getDistanceFunctionFlag(), new WDTWDistance(), subSpace);
        return space;
    }


    public static ParamSpace buildWddtwSpace(Instances data) {
        final ParamSpace space = new ParamSpace();
        final ParamSpace subSpace = new ParamSpace();
        subSpace.add(DTW.getWarpingWindowFlag(), new UniformDistribution(0, 1));
        space.add(DistanceMeasureable.getDistanceFunctionFlag(), new WDDTWDistance(), subSpace);
        return space;
    }
}
