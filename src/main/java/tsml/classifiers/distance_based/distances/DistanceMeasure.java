package tsml.classifiers.distance_based.distances;

import utilities.Debugable;
import utilities.params.ParamHandler;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.neighboursearch.PerformanceStats;

import java.io.Serializable;

public interface DistanceMeasure
    extends Serializable,
            DistanceFunction,
            Debugable,
            ParamHandler
{

    static String getDistanceFunctionFlag() {
        return "d";
    }

    default double getMaxDistance() {
        return Double.POSITIVE_INFINITY;
    }

    void checkData(Instance first, Instance second);

    boolean isSymmetric();

    default double distance(final Instance first, final Instance second) {
        return distance(first, second, getMaxDistance());
    }

    default double distance(final Instance first, final Instance second, final double limit) {
        return distance(first, second, limit, null);
    }

    @Override
    default double distance(final Instance first, final Instance second, final PerformanceStats stats) throws
                                                                                                              Exception {
        return distance(first, second, getMaxDistance(), stats);
    }

    // default implementations of fussy methods around distance measures

    @Override
    default void setAttributeIndices(final String value) {

    }

    @Override
    default String getAttributeIndices() {
        return null;
    }

    @Override
    default void setInvertSelection(final boolean value) {

    }

    @Override
    default boolean getInvertSelection() {
        return false;
    }

    @Override
    default void postProcessDistances(final double[] distances) {

    }

}
