package tsml.classifiers.distance_based.distances;

import utilities.Debugable;
import utilities.ParamHandler;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.neighboursearch.PerformanceStats;

import java.io.Serializable;
import java.util.Enumeration;

public interface DistanceMeasure
    extends Serializable,
            DistanceFunction,
            Debugable,
            ParamHandler
{

    String DISTANCE_FUNCTION_FLAG = "d";
    double MAX_DISTANCE = Double.POSITIVE_INFINITY;

    void checks(Instance first, Instance second);

    boolean isSymmetric();

    // todo list options

    default double distance(final Instance first, final Instance second) {
        return distance(first, second, MAX_DISTANCE);
    }

    default double distance(final Instance first, final Instance second, final double limit) {
        return distance(first, second, limit, null);
    }

    @Override
    default double distance(final Instance first, final Instance second, final PerformanceStats stats) throws
                                                                                                              Exception {
        return distance(first, second, MAX_DISTANCE, stats);
    }

}
