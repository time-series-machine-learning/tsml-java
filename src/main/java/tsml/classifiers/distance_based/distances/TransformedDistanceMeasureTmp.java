package tsml.classifiers.distance_based.distances;
/*

Purpose: automate the transformation of instances before a distance measurement

Contributors: goastler
    
*/

import utilities.cache.CachedFunction;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.neighboursearch.PerformanceStats;

public interface TransformedDistanceMeasureTmp extends DistanceFunction {

    double decoratedDistance(Instance first, Instance second, double limit, PerformanceStats stats);

    CachedFunction<Instance, Instance> getCache();

    void setCache(CachedFunction<Instance, Instance> cache);

    @Override
    default double distance(Instance first, Instance second, double limit, PerformanceStats stats) {
        // take the derivative of both instances
        final CachedFunction<Instance, Instance> cache = getCache();
        Instance transformedFirst = cache.apply(first);
        Instance transformedSecond = cache.apply(second);
        // find the DTW distance between them
        return decoratedDistance(transformedFirst, transformedSecond, limit, stats);
    }
}
