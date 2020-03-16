package tsml.classifiers.distance_based.distances;

import java.io.Serializable;
import tsml.classifiers.distance_based.utils.logging.Debugable;
import tsml.classifiers.distance_based.utils.params.ParamHandler;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.neighboursearch.PerformanceStats;

public interface DistanceMeasureable extends Serializable, DistanceFunction, Debugable, ParamHandler {

    static String getDistanceFunctionFlag() {
        return "d";
    }

//    // the maximum distance the distance measure could produce
//    default double getMaxDistance() {
//        return Double.POSITIVE_INFINITY;
//    }

    // the maximum distance the distance measure could produce
    static double getMaxDistance() {
        return Double.POSITIVE_INFINITY;
    }

    // whether the distance measure is symmetric (i.e. dist from inst A to inst B == dist from inst B to inst A
    default boolean isSymmetric() {
        return true;
    }

    default double distance(final Instance first, final Instance second) {
        return distance(first, second, getMaxDistance());
    }

    default double distance(final Instance first, final Instance second, final double limit) {
        return distance(first, second, limit, null);
    }

    @Override
    default double distance(final Instance first, final Instance second, final PerformanceStats stats)
        throws Exception {
        return distance(first, second, getMaxDistance(), stats);
    }

    String getName();

    // default implementations of fussy methods around distance measures

    @Override
    default String getAttributeIndices() {
        return null;
    }

    @Override
    default void setAttributeIndices(final String value) {

    }

    @Override
    default boolean getInvertSelection() {
        return false;
    }

    @Override
    default void setInvertSelection(final boolean value) {

    }

    @Override
    default void postProcessDistances(final double[] distances) {

    }

}
