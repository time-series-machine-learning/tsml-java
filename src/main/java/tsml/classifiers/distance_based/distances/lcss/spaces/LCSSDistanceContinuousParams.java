package tsml.classifiers.distance_based.distances.lcss.spaces;

import tsml.classifiers.distance_based.distances.lcss.LCSSDistance;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpace;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpaceBuilder;
import tsml.classifiers.distance_based.utils.collections.params.distribution.double_based.UniformDoubleDistribution;
import tsml.data_containers.TimeSeriesInstances;
import utilities.StatisticalUtilities;

import static tsml.classifiers.distance_based.distances.dtw.DTW.WINDOW_SIZE_FLAG;

public class LCSSDistanceContinuousParams implements ParamSpaceBuilder {
    @Override public ParamSpace build(final TimeSeriesInstances data) {
        final double std = StatisticalUtilities.pStdDev(data);
        final ParamSpace subSpace = new ParamSpace();
        subSpace.add(LCSSDistance.EPSILON_FLAG, new UniformDoubleDistribution(0.2 * std, std));
        subSpace.add(WINDOW_SIZE_FLAG, new UniformDoubleDistribution(0d, 1d));
        return subSpace;
    }
}
