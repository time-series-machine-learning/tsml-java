package tsml.classifiers.distance_based.distances.lcss.spaces;

import tsml.classifiers.distance_based.distances.lcss.LCSSDistance;
import tsml.classifiers.distance_based.utils.collections.params.ParamMap;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpace;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpaceBuilder;
import tsml.classifiers.distance_based.utils.collections.params.distribution.double_based.UniformDoubleDistribution;
import tsml.data_containers.TimeSeriesInstances;
import utilities.StatisticalUtilities;

import static tsml.classifiers.distance_based.distances.dtw.DTW.WINDOW_FLAG;

public class LCSSDistanceRestrictedContinuousParams implements ParamSpaceBuilder {
    @Override public ParamSpace build(final TimeSeriesInstances data) {
        final double std = StatisticalUtilities.pStdDev(data);
        final ParamMap subSpace = new ParamMap();
        subSpace.add(LCSSDistance.EPSILON_FLAG, new UniformDoubleDistribution(0.2 * std, std));
        subSpace.add(WINDOW_FLAG, new UniformDoubleDistribution(0d, 0.25));
        return new ParamSpace(subSpace);
    }
}
