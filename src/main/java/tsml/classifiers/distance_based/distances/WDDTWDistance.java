package tsml.classifiers.distance_based.distances;

import tsml.filters.Derivative;
import utilities.cache.CachedFunction;
import weka.core.Instance;
import weka.core.neighboursearch.PerformanceStats;

import static tsml.classifiers.distance_based.distances.DDTWDistance.DERIVATIVE_CACHE;

/**
 * WDDTW distance measure.
 * <p>
 * Contributors: goastler
 */
public class WDDTWDistance extends WDTWDistance implements TransformedDistanceMeasureTmp {

    // Global derivative function which is cached, i.e. if you ask it to convert the same instance twice it will
    // instead fetch from the cache the second time
    private CachedFunction<Instance, Instance> derivativeCache = Derivative.getGlobalCache();

    @Override
    public double decoratedDistance(Instance first, Instance second, double limit, PerformanceStats stats) {
        return super.distance(first, second, limit, stats);
    }

    @Override
    public CachedFunction<Instance, Instance> getCache() {
        return derivativeCache;
    }

    @Override
    public void setCache(CachedFunction<Instance, Instance> cache) {
        derivativeCache = cache;
    }

    @Override
    public double distance(final Instance first,
                           final Instance second,
                           final double limit,
                           final PerformanceStats stats) {
        return TransformedDistanceMeasureTmp.super.distance(first, second, limit, stats);
    }

}
