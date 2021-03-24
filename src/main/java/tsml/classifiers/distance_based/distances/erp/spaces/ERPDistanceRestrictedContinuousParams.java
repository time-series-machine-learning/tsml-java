package tsml.classifiers.distance_based.distances.erp.spaces;

import tsml.classifiers.distance_based.distances.erp.ERPDistance;
import tsml.classifiers.distance_based.utils.collections.params.ParamMap;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpace;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpaceBuilder;
import tsml.classifiers.distance_based.utils.collections.params.distribution.double_based.UniformDoubleDistribution;
import tsml.data_containers.TimeSeriesInstances;
import utilities.StatisticalUtilities;

import static tsml.classifiers.distance_based.distances.dtw.DTW.WINDOW_FLAG;

public class ERPDistanceRestrictedContinuousParams implements ParamSpaceBuilder {
    @Override public ParamSpace build(final TimeSeriesInstances data) {
        final double std = StatisticalUtilities.pStdDev(data);
        final ParamMap subSpace = new ParamMap();
        subSpace.add(ERPDistance.G_FLAG, new UniformDoubleDistribution(0.2 * std, std));
        subSpace.add(WINDOW_FLAG, new UniformDoubleDistribution(0d, 0.25));
        return new ParamSpace(subSpace);
    }
}
