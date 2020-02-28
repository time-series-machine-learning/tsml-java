package tsml.classifiers.distance_based.distances;

import utilities.cache.CachedFunction;
import weka.core.Instance;
import weka.core.neighboursearch.PerformanceStats;

import static tsml.filters.Derivative.INSTANCE_DERIVATIVE_FUNCTION;

/**
 * DDTW distance measure. This is the derivative version of DTW.
 *
 * Contributors: goastler
 */
public class DDTWDistance extends DTWDistance {
    // Global derivative function which is cached, i.e. if you ask it to convert the same instance twice it will instead fetch from the cache the second time
    public static final CachedFunction<Instance, Instance> DERIVATIVE_CACHE = new CachedFunction<>(INSTANCE_DERIVATIVE_FUNCTION);
    // Cache function for taking the derivative. I suggest setting these all to use the same cache if you've got multiple derivative based distance measures
    private CachedFunction<Instance, Instance> derivativeCache = DERIVATIVE_CACHE;

    public CachedFunction<Instance, Instance> getDerivativeCache() {
        return derivativeCache;
    }

    public void setDerivativeCache(final CachedFunction<Instance, Instance> derivativeCache) {
        this.derivativeCache = derivativeCache;
    }

    @Override
    public double distance(final Instance first,
                           final Instance second,
                           final double limit,
                           final PerformanceStats stats) {
        // take the derivative of both instances
        Instance transformedFirst = derivativeCache.apply(first);
        Instance transformedSecond = derivativeCache.apply(second);
        // find the DTW distance between them
        return super.distance(transformedFirst, transformedSecond, limit, stats);
    }

    public DDTWDistance() {
        super();
    }

    public DDTWDistance(int warpingWindow) {
        super(warpingWindow);
    }

}
