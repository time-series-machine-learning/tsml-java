package tsml.classifiers.distance_based.distances.lcss.spaces;

import tsml.classifiers.distance_based.distances.lcss.LCSSDistance;
import tsml.classifiers.distance_based.utils.collections.params.ParamMap;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpace;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpaceBuilder;
import tsml.data_containers.TimeSeriesInstances;
import utilities.StatisticalUtilities;

import java.util.List;

import static tsml.classifiers.distance_based.distances.dtw.DTW.WINDOW_FLAG;
import static utilities.ArrayUtilities.range;
import static utilities.ArrayUtilities.unique;

public class LCSSDistanceParams implements ParamSpaceBuilder {
    @Override public ParamSpace build(final TimeSeriesInstances data) {
        double std = StatisticalUtilities.pStdDev(data);
        double stdFloor = std * 0.2;
        double[] epsilonValues = range(stdFloor, std, 10);
        double[] deltaValues = range(0d, 0.25, 10);
        List<Double> epsilonValuesUnique = unique(epsilonValues);
        List<Double> deltaValuesUnique = unique(deltaValues);
        ParamMap params = new ParamMap();
        params.add(LCSSDistance.EPSILON_FLAG, epsilonValuesUnique);
        params.add(WINDOW_FLAG, deltaValuesUnique);
        return new ParamSpace(params);
    }
}
