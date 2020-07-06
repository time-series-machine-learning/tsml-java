package tsml.classifiers.distance_based.distances.erp;

import com.beust.jcommander.internal.Lists;
import tsml.classifiers.distance_based.distances.DistanceMeasure;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpace;
import tsml.classifiers.distance_based.utils.collections.params.distribution.double_based.UniformDoubleDistribution;
import tsml.classifiers.distance_based.utils.collections.params.distribution.int_based.UniformIntDistribution;
import utilities.StatisticalUtilities;
import weka.core.Instances;

import java.util.List;

import static tsml.classifiers.distance_based.utils.collections.CollectionUtils.newArrayList;
import static utilities.ArrayUtilities.range;
import static utilities.ArrayUtilities.unique;

public class ERPDistanceConfigs {
    public static ParamSpace buildErpSpace(Instances instances) {
        return new ParamSpace().add(DistanceMeasure.DISTANCE_MEASURE_FLAG, newArrayList(new ERPDistance()),
                buildErpParams(instances));
    }

    public static ParamSpace buildErpParams(Instances instances) {
        double std = StatisticalUtilities.pStdDev(instances);
        double stdFloor = std * 0.2;
        int[] bandSizeValues = range(0, (instances.numAttributes() - 1) / 4, 10);
        double[] penaltyValues = range(stdFloor, std, 10);
        List<Double> penaltyValuesUnique = unique(penaltyValues);
        List<Integer> bandSizeValuesUnique = unique(bandSizeValues);
        ParamSpace params = new ParamSpace();
        params.add(ERPDistance.WINDOW_SIZE_FLAG, bandSizeValuesUnique);
        params.add(ERPDistance.G_FLAG, penaltyValuesUnique);
        return params;
    }

    public static ParamSpace buildErpParamsContinuous(Instances data) {
        final double std = StatisticalUtilities.pStdDev(data);
        final ParamSpace subSpace = new ParamSpace();
//        subSpace.add(ERPDistance.G_FLAG, new DoubleDistribution(0,1) {
//
//            @Override
//            public Double sample() {
//                return getRandom().nextDouble() * 0.8 * std + 0.2 * std;
//            }
//        });
        subSpace.add(ERPDistance.G_FLAG, new UniformDoubleDistribution(0.2 * std, std));
        // pf implements this as randInt(len / 4 + 1), so range is from 0 to len / 4 inclusively
        // above doesn't consider class value, so -1 from len
        subSpace.add(ERPDistance.WINDOW_SIZE_FLAG, new UniformIntDistribution(0,
            (data.numAttributes() - 1) / 4));
        return subSpace;
    }

    public static ParamSpace buildErpSpaceContinuous(Instances data) {
        final ParamSpace space = new ParamSpace();
        space.add(DistanceMeasure.DISTANCE_MEASURE_FLAG, Lists.newArrayList(new ERPDistance()),
                  buildErpParamsContinuous(data));
        return space;
    }
}
