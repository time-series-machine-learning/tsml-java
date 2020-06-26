package tsml.classifiers.distance_based.distances;

import java.io.Serializable;
import tsml.classifiers.distance_based.utils.collections.params.ParamHandler;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.neighboursearch.PerformanceStats;

public interface DistanceMeasure extends Serializable, DistanceFunction, ParamHandler {

    String DISTANCE_MEASURE_FLAG = "d";

    // the maximum distance the distance measure could produce
    double getMaxDistance();

    // whether the distance measure is symmetric (i.e. dist from inst A to inst B == dist from inst B to inst A
    boolean isSymmetric();

    double distance(final Instance a, final Instance b);

    double distance(final Instance a, final Instance b, final double limit);

    @Override
    double distance(final Instance a, final Instance b, final PerformanceStats stats)
        throws Exception;

    String getName();

    void setName(String name);

    // the fit function
    void setInstances(Instances data);


    static String getName(DistanceFunction df) {
        if(df instanceof DistanceMeasure) {
            return ((DistanceMeasure) df).getName();
        } else {
            return df.getClass().getSimpleName();
        }
    }
}
