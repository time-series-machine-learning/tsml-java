package tsml.classifiers.distance_based.distances;

import java.io.Serializable;

import tsml.classifiers.distance_based.utils.collections.params.ParamHandler;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import tsml.data_containers.utilities.Converter;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.neighboursearch.PerformanceStats;

public interface DistanceMeasure extends Serializable, ParamHandler {

    String DISTANCE_MEASURE_FLAG = "d";

    // whether the distance measure is symmetric (i.e. dist from inst A to inst B == dist from inst B to inst A
    default boolean isSymmetric() {
        return true;
    }

    default double distance(final Instance a, final Instance b) {
        return distance(a, b, Double.POSITIVE_INFINITY);
    }
    
    /**
     * Override this distance func
     * @param a
     * @param b
     * @param limit
     * @return
     */
    default double distance(final Instance a, final Instance b, final double limit) {
        return distance(Converter.fromArff(a), Converter.fromArff(b), limit);
    }

    default double distance(final TimeSeriesInstance a, final TimeSeriesInstance b) {
        return distance(a, b, Double.POSITIVE_INFINITY);
    }

    /**
     * Or override this distance func
     * @param a
     * @param b
     * @param limit
     * @return
     */
    default double distance(final TimeSeriesInstance a, final TimeSeriesInstance b, double limit) {
        return distance(Converter.toArff(a), Converter.toArff(b), limit);
    }
    
    default String getName() {
        return getClass().getSimpleName();
    }

    default void buildDistanceMeasure(TimeSeriesInstances data) {
        
    }
    
    default void buildDistanceMeasure(Instances data) {
        buildDistanceMeasure(Converter.fromArff(data));
    }
    
    default DistanceFunction asDistanceFunction() {
        return new DistanceFunctionAdapter(this);
    }
    
    static DistanceMeasure asDistanceMeasure(DistanceFunction df) {
        return new DistanceMeasureAdapter(df);
    }
}
